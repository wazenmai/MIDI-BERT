import numpy as np
import math
import sys
import os
import copy
import random
import shutil

from transformers import BertModel, BertConfig, AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import pickle
import argparse

'''
BertConfig {
    "attention_probs_dropout_prob": 0.1,
    "gradient_checkpointing": false,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "relative",
    "transformers_version": "4.4.1",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 30522
}
'''
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict-file', type=str, default='dict/CP.pkl')
    parser.add_argument('--name', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--mask-percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--init-lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    return args


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# BERT model: similar approach to "felix"
class BertForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super(BertForPredictingMiddleNotes, self).__init__()
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.bertConfig = bertConfig
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        # token types: [Tempo, Bar, Position, Pitch, Duration, Velocity]
        self.n_tokens = []      #[3,18,88,66]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        # 2 

        #self.eos_word = torch.Tensor([self.e2w[etype]['%s <EOS>' % etype] for etype in self.e2w]).long().to(device)
        #self.sos_word = torch.Tensor([self.e2w[etype]['%s <SOS>' % etype] for etype in self.e2w]).long().to(device)

        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types to feed into transformer-XL
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(bertConfig.d_model, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

    def forward(self, input_ids, attn_mask=None):
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask)
        y = y.last_hidden_state         # (batch_size, seq_len, 768)

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys


    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_rand_tok(self):
        c1,c2,c3,c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array([random.choice(range(c1)),random.choice(range(c2)),random.choice(range(c3)),random.choice(range(c4))])


def get_mask_ind(Lseq):
    mask_ind = random.sample(Lseq, round(args.max_seq_len * args.mask_percent))
    mask80 = random.sample(mask_ind, round(len(mask_ind)*0.8))
    left = list(set(mask_ind)-set(mask80))
    rand10 = random.sample(left, round(len(mask_ind)*0.1))
    cur10 = list(set(left)-set(rand10))
    return mask80, rand10, cur10



def training(model, training_data, optimizer, e2w, Lseq):
    num_batches = len(training_data) // (args.batch_size)
    total_losses = 0
    total_acc = [0,0,0,0]
    for train_iter in range(num_batches):
        ori_seq_batch = training_data[train_iter*args.batch_size:(train_iter+1)*args.batch_size]

        input_ids = copy.deepcopy(ori_seq_batch)
        loss_mask = torch.zeros(args.batch_size, args.max_seq_len)

        for b in range(args.batch_size):
            # get index for masking
            mask80, rand10, cur10 = get_mask_ind(Lseq)
            # apply mask, random, remain current token
            for i in mask80:
                input_ids[b][i] = model.mask_word_np
                loss_mask[b][i] = 1 
            for i in rand10:
                input_ids[b][i] = model.get_rand_tok()
                loss_mask[b][i] = 1 
            for i in cur10:
                loss_mask[b][i] = 1 
        
        input_ids = input_ids.astype(float)
        input_ids = torch.from_numpy(input_ids).to(device).long()   # (4,512,4)
        
        loss_mask = loss_mask.to(device).long()     # (4,512)

        # avoid attend to pad word
        attn_mask = (input_ids[:, :, 0] != model.bar_pad_word).float()   # (4,512)

        y = model.forward(input_ids, attn_mask)

        # get the most likely choice with max
        outputs = []
        for i, etype in enumerate(e2w):
            output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
            outputs.append(output)
        outputs = np.stack(outputs, axis=-1)    
        outputs = outputs.astype(float)
        outputs = torch.from_numpy(outputs).to(device).long()   # (4,512)

        # accuracy
        ori_seq_batch = ori_seq_batch.astype(float)
        ori_seq_batch = torch.from_numpy(ori_seq_batch).to(device).long()
        
        all_acc = []
        for i in range(4):
            acc = torch.sum((ori_seq_batch[:,:,i] == outputs[:,:,i]).float() * loss_mask)
            acc /= torch.sum(loss_mask)
            all_acc.append(acc)
        total_acc = [sum(x) for x in zip(total_acc, all_acc)]

        # reshape (b, s, f) -> (b, f, s)
        for i, etype in enumerate(e2w):
            #print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
            y[i] = y[i][:, ...].permute(0, 2, 1)

        # calculate losses
        losses = []
        for i, etype in enumerate(e2w):
            losses.append(model.compute_loss(y[i], ori_seq_batch[..., i].to(device), loss_mask.to(device)))
        total_loss = sum(losses) / len(e2w)

        # udpate
        model.zero_grad()
        total_loss.backward()
        clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        # acc
        accs = list(map(float, all_acc))
        sys.stdout.write('{}/{} | Loss: {:06f} | loss: {:04f}, {:04f}, {:04f}, {:04f} | acc: {:04f}, {:04f}, {:04f}, {:04f} \r'.format(
            train_iter, num_batches, total_loss, *losses, *accs))

        losses = list(map(float, losses))
        total_losses += total_loss.item()

    return total_losses/num_batches, [round(x.item()/num_batches,4) for x in total_acc]


def validating(model, valid_data, Lseq):
    num_batches = len(valid_data) // (args.batch_size)
    total_losses = 0
    total_acc = [0,0,0,0]
    
    model.eval()
    with torch.no_grad():
        for iter in range(num_batches):
            ori_seq_batch = valid_data[iter*args.batch_size:(iter+1)*args.batch_size]

            input_ids = copy.deepcopy(ori_seq_batch)
            loss_mask = torch.zeros(args.batch_size, args.max_seq_len)

            for b in range(args.batch_size):
                # get index for masking
                mask80, rand10, cur10 = get_mask_ind(Lseq)
                # apply mask, random, remain current token
                for i in mask80:
                    input_ids[b][i] = model.mask_word_np
                    loss_mask[b][i] = 1 
                for i in rand10:
                    input_ids[b][i] = model.get_rand_tok()
                    loss_mask[b][i] = 1 
                for i in cur10:
                    loss_mask[b][i] = 1 
            
            input_ids = input_ids.astype(float)
            input_ids = torch.from_numpy(input_ids).to(device).long()   # (4,512,4)
            
            loss_mask = loss_mask.to(device).long()     # (4,512)

            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != model.bar_pad_word).float()   # (4,512)

            y = model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)    
            outputs = outputs.astype(float)
            outputs = torch.from_numpy(outputs).to(device).long()   # (4,512)

            # accuracy
            ori_seq_batch = ori_seq_batch.astype(float)
            ori_seq_batch = torch.from_numpy(ori_seq_batch).to(device).long()
            
            all_acc = []
            for i in range(4):
                acc = torch.sum((ori_seq_batch[:,:,i] == outputs[:,:,i]).float() * loss_mask)
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(e2w):
                #print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses = []
            for i, etype in enumerate(e2w):
                losses.append(model.compute_loss(y[i], ori_seq_batch[..., i].to(device), loss_mask.to(device)))
            total_loss = sum(losses) / len(e2w)

            # acc
            accs = list(map(float, all_acc))
            sys.stdout.write('{}/{} | Loss: {:06f} | loss: {:04f}, {:04f}, {:04f}, {:04f} | acc: {:04f}, {:04f}, {:04f}, {:04f} \r'.format(
                iter, num_batches, total_loss, *losses, *accs))

            losses = list(map(float, losses))
            total_losses += total_loss.item()
            
        return total_losses/num_batches, [round(x.item()/num_batches,4) for x in total_acc]

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    best_mdl = filename.split('.')[0]+'_best.ckpt'
    if is_best:
        shutil.copyfile(filename, best_mdl)


if __name__ == '__main__':
    # get arguments
    args = get_args()

    cuda_str = 'cuda:' + str(args.cuda)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type="relative_key_query",
                               hidden_size=args.hs)

    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
        #print(e2w)

    print('\nInitializing model...')
    model = BertForPredictingMiddleNotes(configuration, e2w, w2e).to(device)
    print(model)

    print('\nLoading data...')
    # load data
    POP909_path = '/home/yh1488/NAS-189/home/CP_data/POP909cp.npy'
    POP_data = np.load(POP909_path, allow_pickle=True)
#    POP909_train, _, _ = np.split(POP_data, [int(.8 * len(POP_data)), int(.9 * len(POP_data))])
    print('POP909:', POP_data.shape)

    composer_path = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_train.npy'
    composer_train = np.load(composer_path, allow_pickle=True)
    composer_path = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_valid.npy'
    composer_valid = np.load(composer_path, allow_pickle=True)
    composer_path = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/composer_cp_test.npy'
    composer_test = np.load(composer_path, allow_pickle=True)
    print('composer: ({}, {}, {})'.format(composer_train.shape[0]+composer_valid.shape[0]+composer_test.shape[0], composer_train.shape[1], composer_train.shape[2]))

    remi1700_path = '/home/yh1488/NAS-189/home/CP_data/ai17k.npy'
    remi1700 = np.load(remi1700_path, allow_pickle=True)
    print('remi1700:', remi1700.shape)

    ASAP_path = '/home/yh1488/NAS-189/homes/wazenmai/MIDI-BERT/for_wazenmai/prepare_CP/ASAP_CP.npy'
    ASAP = np.load(ASAP_path, allow_pickle=True)
    print('ASAP:', ASAP.shape)

    emopia_path = '/home/yh1488/NAS-189/homes/wazenmai/MIDI-BERT/for_wazenmai/prepare_CP/Emopia_CP.npy'
    emopia = np.load(emopia_path, allow_pickle=True)
    print('emopia:', emopia.shape)
    
    training_data = np.concatenate((composer_train, remi1700, composer_valid, POP_data, composer_test, ASAP, emopia), axis=0) 
    #training_data = remi1700
    print('> all training data:', training_data.shape)

    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data)*0.85)
    X_train, X_val = training_data[:split], training_data[split:]

    print('\nStart pre-training')
    #/NAS-189/home/BERT/cp_result/pretrain
    save_dir = '/home/yh1488/NAS-189/homes/yh1488/BERT/cp_result/pretrain/'
    os.makedirs(save_dir, exist_ok=True)
    #path_saved_ckpt = os.path.join(save_dir, 'epoch')

    Lseq = [i for i in range(args.max_seq_len)]
    optimizer = AdamW(model.parameters(), lr=args.init_lr, weight_decay=0.01)

    best_acc, bad_cnt = 0, 0
    filename = save_dir + 'model-' + args.name + '.ckpt'
    for epoch in range(args.epochs):
        if bad_cnt >= 20:
            print('valid acc not improving for 20 epochs')
            break
        train_loss, train_acc = training(model, X_train, optimizer, e2w, Lseq)
        valid_loss, valid_acc = validating(model, X_val, Lseq)
        weighted_score = [x*y for (x,y) in zip(valid_acc, model.n_tokens)]
        avg_acc = sum(weighted_score)/sum(model.n_tokens)
        
        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)
        
        if is_best:
            bad_cnt = 0
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {:.4f} | Train acc: {} | Valid Loss: {:.4f} | Valid acc: {}'.format(
            epoch, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer' : optimizer.state_dict(),
            }, is_best, filename)

        '''if 0.1 < valid_loss <= 0.5:
            fn = int(loss_val * 100)
            if fn % 2 == 0:
                torch.save(model.state_dict(), path_saved_ckpt + str(epoch) + '_loss' + str(fn) + '.ckpt')
        '''
