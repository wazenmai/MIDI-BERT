import numpy as np
import math
import sys
import os
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
torch.manual_seed(2021)

def get_args():
    parser = argparse.ArgumentParser(description='')
    
    ### mode ###
    parser.add_argument('-d', '--dataset', choices=['pop909', 'composer'], required=True)

    ### path setup ###
    parser.add_argument('--dict-file', type=str, default='dict/remi.pkl')
    parser.add_argument('--data-file', type=str, default='/home/yh1488/NAS-189/home/remi_data/POP909remi.npy')
    parser.add_argument('--ckpt-path', type=str, default='/home/yh1488/NAS-189/home/BERT/remi_result/pretrain/model-final_best.ckpt')
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
        '''self.n_tokens = []      #[3,18,88,66]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]'''
        self.n_token = len(e2w)
        self.emb_size = 256
        self.e2w = e2w
        self.w2e = w2e
        self.index_layer = index_layer

        # for deciding whether the current input_ids is a <PAD> token
#        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']        # 2 
        self.pad_word = self.e2w['Pad_None']
        self.mask_word = self.e2w['Mask_None']

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = Embeddings(self.n_token, self.emb_size)

        # linear layer to merge embeddings from different token types to feed into transformer-XL
        self.in_linear = nn.Linear(self.emb_size, bertConfig.d_model)

        # proj: project embeddings to logits for prediction
        '''class_num = 4 if args.task=="melody" else 5
        self.proj_linear = nn.Linear(bertConfig.d_model, class_num)'''

    def forward(self, input_id, attn_mask=None):
        # convert input_ids into embeddings and merge them through linear layer
        '''embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)'''
        emb = self.word_emb(input_id)
        emb_linear = self.in_linear(emb)

        # feed to bert 
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=True)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[self.index_layer]

        # convert embeddings back to logits for prediction
        #y = self.proj_linear(y)
        return y


def extracting(model, data):
    num_batches = len(data) // (args.batch_size)
    
    model.eval()
    outputs = torch.zeros(data.shape[0], data.shape[1], 768)
    with torch.no_grad():
        for ft_iter in range(num_batches):
            input_ids = data[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size]
            input_ids = input_ids.astype(float)
            input_ids = torch.from_numpy(input_ids).to(device).long()           # (4,512,4)
            # avoid attend to pad word
            pad_tensor = model.pad_word * torch.ones(args.batch_size, args.max_seq_len)
            pad_tensor = pad_tensor.to(device).long()
            attn_mask = (pad_tensor != input_ids).float()       # (batch,512)
            
            y = model.forward(input_ids, attn_mask)
            outputs[ft_iter*args.batch_size:(ft_iter+1)*args.batch_size] = y

        return outputs


def load_pop909():
    pop909_path='/home/yh1488/NAS-189/home/remi_data/POP909remi.npy'
    finetune_data = np.load(pop909_path, allow_pickle=True)
    X_train, X_val, X_test = np.split(finetune_data, [int(.8 * len(finetune_data)), int(.9 * len(finetune_data))])
    print('   POP train', X_train.shape)
    print('   POP valid', X_val.shape)
    print('   POP test', X_test.shape)
    return X_train, X_val, X_test


def load_composer():
    root = '/home/yh1488/NAS-189/homes/wazenmai/datasets/MIDI-BERT/composer_dataset/remi/'
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

    print('Loading data...')
    if args.dataset == 'pop909':
        X_train, X_val, X_test = load_pop909()
    elif args.dataset == 'composer':
        X_train, X_val, X_test = load_composer()

    # shuffle 
#    index = np.arange(len(X_train))
#    np.random.shuffle(index)
#    X_train = X_train[index]
#    y_train = y_train[index]

    print('\nInitializing model...')
    index_layer = int(args.layer)-13
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type="relative_key_query")
    model = BertForPredictingMiddleNotes(configuration, e2w, w2e, index_layer).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'), strict=False)
    
    print('\nExtracting...')
    print('   Embedding layer {}, which is hidden [{}]\n'.format(args.layer, index_layer))

    root = '/home/yh1488/NAS-189/home/BERT/remi_embed/'
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
   
