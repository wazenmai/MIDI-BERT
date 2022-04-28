import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import tqdm
import sys
import shutil
import copy

from MidiBERT.model import MidiBert
from MidiBERT.modelLM import MidiBertLM


class BERTTrainer:
    def __init__(self, midibert: MidiBert, train_dataloader, valid_dataloader, 
                lr, batch, max_seq_len, mask_percent, cpu, cuda_devices=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.midibert = midibert        # save this for ckpt
        self.model = MidiBertLM(midibert).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
    
    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_mask_ind(self):
        mask_ind = random.sample(self.Lseq, round(self.max_seq_len * self.mask_percent))
        mask80 = random.sample(mask_ind, round(len(mask_ind)*0.8))
        left = list(set(mask_ind)-set(mask80))
        rand10 = random.sample(left, round(len(mask_ind)*0.1))
        cur10 = list(set(left)-set(rand10))
        return mask80, rand10, cur10


    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        valid_loss, valid_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
        return valid_loss, valid_acc

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_losses = [0]*len(self.midibert.e2w), 0
        
        for ori_seq_batch in pbar:
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.to(self.device)  # (batch, seq_len, 4) 
            input_ids = copy.deepcopy(ori_seq_batch)
            loss_mask = torch.zeros(batch, max_seq_len)
            
            for b in range(batch):
                # get index for masking
                mask80, rand10, cur10 = self.get_mask_ind()
                # apply mask, random, remain current token
                for i in mask80:
                    mask_word = torch.tensor(self.midibert.mask_word_np).to(self.device)
                    input_ids[b][i] = mask_word 
                    loss_mask[b][i] = 1 
                for i in rand10:
                    rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                    input_ids[b][i] = rand_word 
                    loss_mask[b][i] = 1 
                for i in cur10:
                    loss_mask[b][i] = 1 
            
            loss_mask = loss_mask.to(self.device)      

            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float().to(self.device)   # (batch, seq_len)
            
            y = self.model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.midibert.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)    
            outputs = torch.from_numpy(outputs).to(self.device)   # (batch, seq_len)

            # accuracy
            all_acc = []
            for i in range(4):
                acc = torch.sum((ori_seq_batch[:,:,i] == outputs[:,:,i]).float() * loss_mask)
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                #print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.midibert.e2w):
                n_tok.append(len(self.midibert.e2w[etype]))
                losses.append(self.compute_loss(y[i], ori_seq_batch[..., i], loss_mask))
            total_loss_all = [x*y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all)/sum(n_tok)   # weighted

            # udpate only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)
                self.optim.step()

            # acc
            accs = list(map(float, all_acc))
            sys.stdout.write('Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \r'.format(
                total_loss, *losses, *accs)) 

            losses = list(map(float, losses))
            total_losses += total_loss.item()
        
        return round(total_losses/len(training_data),3), [round(x.item()/len(training_data),3) for x in total_acc]

    def save_checkpoint(self, epoch, best_acc, valid_acc, 
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.midibert.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer' : self.optim.state_dict()
        }

        torch.save(state, filename)

        best_mdl = filename.split('.')[0]+'_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)

