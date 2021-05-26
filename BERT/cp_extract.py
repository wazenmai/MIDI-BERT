import numpy as np
import math
import sys
import os
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
torch.manual_seed(2021)

def get_args():
    parser = argparse.ArgumentParser(description='')
    
    ### mode ###
    parser.add_argument('-d', '--dataset', choices=['pop909', 'composer'], required=True)
    
    ### path setup ###
    parser.add_argument('--dict-file', type=str, default='dict/CP.pkl')
    parser.add_argument('--ckpt-path', type=str, default='/home/yh1488/NAS-189/home/BERT/cp_result/pretrain/model-final_best.ckpt')
    parser.add_argument('--layer', type=str, default='12', required=True)

    ### parameter setting ###
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
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
    def __init__(self, bertConfig, e2w, w2e, index_layer):
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
        self.index_layer = index_layer

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
        '''self.proj = []
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(bertConfig.d_model, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)'''

    def forward(self, input_ids, attn_mask=None):
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=True)
        #print(len(y.hidden_states)) # 13
        y = y.hidden_states[self.index_layer]
        
#        y = y.last_hidden_state         # (batch_size, seq_len, 768)

        return y
        # convert embeddings back to logits for prediction
        '''ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys'''


def extracting(model, data):
    num_batches = len(data) // (args.batch_size)
    
    model.eval()
    outputs = torch.zeros(data.shape[0], data.shape[1], 768) # hidden_size=768
    with torch.no_grad():
        for ft_iter in range(num_batches):
            input_ids = data[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size]
            input_ids = input_ids.astype(float)
            input_ids = torch.from_numpy(input_ids).to(device).long()           # (4,512,4)
            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != model.bar_pad_word).float()       # (4,512)

            y = model.forward(input_ids, attn_mask)
            outputs[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size] = y

        return outputs


def load_pop909():
    pop909_path='/home/yh1488/NAS-189/home/CP_data/POP909cp.npy'
    finetune_data = np.load(pop909_path, allow_pickle=True)
    X_train, X_val, X_test = np.split(finetune_data, [int(.8 * len(finetune_data)), int(.9 * len(finetune_data))])
    print('   POP train', X_train.shape)
    print('   POP valid', X_val.shape)
    print('   POP test', X_test.shape)
    return X_train, X_val, X_test


def load_composer():
    root = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/CP/'
    X_train = np.load(root+'composer_cp_train.npy', allow_pickle=True)
    X_val = np.load(root+'composer_cp_valid.npy', allow_pickle=True)
    X_test = np.load(root+'composer_cp_test.npy', allow_pickle=True)
    
    print('   composer train', X_train.shape)
    print('   composer valid', X_val.shape)
    print('   composer test', X_test.shape)
    return X_train, X_val, X_test


if __name__ == '__main__':
    # get arguments
    args = get_args()

    cuda_str = 'cuda:' + str(args.cuda)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print('device:', device)


    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
        #print(e2w)

    print('\nLoading data...')

    if args.dataset == 'pop909':
        X_train, X_val, X_test = load_pop909()
    elif args.dataset == 'composer':
        X_train, X_val, X_test = load_composer()
    
    # shuffle 
#    index = np.arange(len(X_train))
#    np.random.shuffle(index)
#    X_train = X_train[index]

    print('\nInitializing model...')
    # 12-> -1, 8 -> -5
    index_layer = int(args.layer)-13
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type="relative_key_query")
    model = BertForPredictingMiddleNotes(configuration, e2w, w2e, index_layer).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'), strict=False)
    #print(model)
    root = '/home/yh1488/NAS-189/home/BERT/cp_embed/'
    print('\nExtracting...')
    print('   Embedding layer {}, which is hidden [{}]\n'.format(args.layer, index_layer))

    name_root = 'POP' if args.dataset == 'pop909' else 'Composer'
    outputs = extracting(model, X_train)
    print(name_root,'train embedding shape', outputs.shape)
    np.save(root + name_root+'train_'+args.layer+'.npy', np.array(outputs))

    outputs = extracting(model, X_val)
    print(name_root, 'valid embedding shape', outputs.shape)
    np.save(root + name_root+'valid_'+args.layer+'.npy', np.array(outputs))

    outputs = extracting(model, X_test)
    print(name_root,'test embedding shape', outputs.shape)
    np.save(root + name_root+'test_'+args.layer+'.npy', np.array(outputs))
        
