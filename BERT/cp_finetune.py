import numpy as np
import math
import sys
import os
import random
import shutil

from transformers import BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup

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
    ### mode ###
    parser.add_argument('-t', '--task', choices=['melody', 'velocity'], required=True)

    ### path setup ###
    parser.add_argument('--dict-file', type=str, default='dict/CP.pkl')
    parser.add_argument('--data-file', type=str, default='/home/yh1488/NAS-189/home/CP_data/POP909cp.npy')
    parser.add_argument('--ans-file', type=str)
    parser.add_argument('--ckpt-path', type=str, default='model_best.ckpt')

    ### parameter setting ###
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--hidden-size', type=int, default=384)
    parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--epochs', type=int, default=4, help='number of finetuning epochs')
    parser.add_argument('--init-lr', type=float, default=2e-5, help='initial learning rate')    # 5e-5
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()
    
    if args.task == 'melody':
        args.ans_file ='/home/yh1488/NAS-189/home/CP_data/POP909cp_melans.npy'
    elif args.task == 'velocity':
        args.ans_file ='/home/yh1488/NAS-189/home/CP_data/POP909cp_velans.npy'

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
    def __init__(self, bertConfig, e2w, w2e, freeze=False):
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

        self.proj = []
        self.all_token = 0
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(bertConfig.d_model, self.n_tokens[i]))
            self.all_token += self.n_tokens[i]
        self.proj = nn.ModuleList(self.proj)

        # proj: project embeddings to logits for prediction
        class_num = 4 if args.task=="melody" else 5
        self.proj_linear = nn.Linear(self.all_token, class_num)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

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

        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))

        # convert embeddings back to logits for prediction
        ys = torch.cat([*ys], dim=-1)
        y = self.proj_linear(ys)
        return y


    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


def training(model, train_data, ans_data, optimizer, scheduler):
    num_batches = len(train_data) // (args.batch_size)
    total_loss, total_acc = 0, 0
    
    model.train()

    for ft_iter in range(num_batches):
        input_ids = train_data[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size]
        input_ids = input_ids.astype(float)
        input_ids = torch.from_numpy(input_ids).to(device).long()           # (4,512,4)
        ans = ans_data[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size]
        ans = ans.astype(float)
        ans = torch.from_numpy(ans).to(device).long()                       # (4,512)

        # avoid attend to pad word
        attn_mask = (input_ids[:, :, 0] != model.bar_pad_word).float()       # (4,512)

        y = model.forward(input_ids, attn_mask)

        # get the most likely choice with max
        output = np.argmax(y.cpu().detach().numpy(), axis=-1)               
        output = output.astype(float)
        output = torch.from_numpy(output).to(device).long()                 # (4,512)

        # accuracy
        acc = torch.sum((ans == output).float() * attn_mask)
        acc /= torch.sum(attn_mask)
        total_acc += acc

        # reshape (batch, seq_len, class) -> (batch, class, seq_len)
        y = y[:, ...].permute(0, 2, 1)

        # calculate losses
        loss = model.compute_loss(y, ans.to(device), attn_mask.to(device))

        # udpate
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()

        # acc
        sys.stdout.write('{}/{} | Loss: {:06f} | acc: {:06f} \r'.format(
            ft_iter, num_batches, loss, acc))

        total_loss += loss.item()
    return total_loss/num_batches, total_acc/num_batches


def validating(model, valid_data, ans_data):
    num_batches = len(valid_data) // (args.batch_size)
    total_loss, total_acc = 0, 0
    
    model.eval()
    with torch.no_grad():
        for ft_iter in range(num_batches):
            input_ids = valid_data[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size]
            input_ids = input_ids.astype(float)
            input_ids = torch.from_numpy(input_ids).to(device).long()           # (4,512,4)
            ans = ans_data[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size]
            ans = ans.astype(float)
            ans = torch.from_numpy(ans).to(device).long()                       # (4,512)

            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != model.bar_pad_word).float()       # (4,512)

            y = model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            output = np.argmax(y.cpu().detach().numpy(), axis=-1)               
            output = output.astype(float)
            output = torch.from_numpy(output).to(device).long()                 # (4,512)

            # accuracy
            acc = torch.sum((ans == output).float() * attn_mask)
            acc /= torch.sum(attn_mask)
            total_acc += acc

            # reshape (batch, seq_len, class) -> (batch, class, seq_len)
            y = y[:, ...].permute(0, 2, 1)

            # calculate losses
            loss = model.compute_loss(y, ans.to(device), attn_mask.to(device))

            # acc
            sys.stdout.write('{}/{} | Loss: {:06f} | acc: {:06f} \r'.format(
                ft_iter, num_batches, loss, acc))

            total_loss += loss.item()
        return total_loss/num_batches, total_acc/num_batches

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    best_mdl = filename.split('_')[0]+'_best.ckpt'
    if is_best:
        shutil.copyfile(filename, best_mdl)


if __name__ == '__main__':
    # get arguments
    args = get_args()

    cuda_str = 'cuda:' + str(args.cuda)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # set seed
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)


    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
        #print(e2w)

    print('Loading data...')
    finetune_data = np.load(args.data_file, allow_pickle=True)
    ans_data = np.load(args.ans_file, allow_pickle=True)
    X_train, X_val, X_test = np.split(finetune_data, [int(.8 * len(finetune_data)), int(.9 * len(finetune_data))])
    y_train, y_val, y_test = np.split(ans_data, [int(.8 * len(ans_data)), int(.9 * len(ans_data))])
    
    # shuffle 
    index = np.arange(len(X_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    print('Initializing model...')
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type="relative_key_query",
                               hidden_size=args.hidden_size)

    model = BertForPredictingMiddleNotes(configuration, e2w, w2e).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'), strict=False)
    
    print('Start fine-tuning on {} task'.format(args.task))
   
    save_dir = '/home/yh1488/NAS-189/home/BERT/cp_result'
#    os.makedirs(save_dir, exist_ok=True)
#    path_saved_ckpt = os.path.join(save_dir, 'epoch')

    optimizer = AdamW(model.parameters(), lr=args.init_lr, weight_decay=0.01)
    num_batches = len(X_train) // (args.batch_size)
    total_steps = num_batches * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
            num_warmup_steps=0, num_training_steps=total_steps)

    best_acc = 0
    filename = save_dir + '/' + args.task + 'FT_model.ckpt'
    
    for epoch in range(args.epochs):
        train_loss, train_acc = training(model, X_train, y_train, optimizer, scheduler)
        valid_loss, valid_acc = validating(model, X_val, y_val)
        
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
            epoch, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        '''save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, filename)
        '''
