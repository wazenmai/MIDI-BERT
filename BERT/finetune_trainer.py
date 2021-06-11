import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

import shutil
import numpy as np
import copy
import random
import tqdm
import sys

from finetune_model import FinetuneModel

class FinetuneTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, test_dataloader, 
                lr, class_num, hs, with_cuda: bool=True, cuda_devices=None, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.midibert = midibert

        if model != None:    # load model
            print('load fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model')
            self.model = FinetuneModel(self.midibert, class_num, hs).to(self.device)
    
        if torch.cuda.device_count() > 1:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader
        
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        self.loss_func = nn.CrossEntropyLoss(reduction='none')
    
    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        valid_loss, valid_acc = self.iteration(self.valid_data, train=False)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        test_loss, test_acc = self.iteration(self.test_data, train=False)
        return test_loss, test_acc

    def iteration(self, training_data, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_loss = 0, 0
        
        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]
            #x, y = x.to(self.device).float(), y.to(self.device)    # no bert: (batch, 512, 4)
            x, y = x.to(self.device), y.to(self.device)

            # avoid attend to pad word
            attn = (y != 0).float().to(self.device)   # (batch,512)

            y_hat = self.model.forward(x, attn)

            # get the most likely choice with max
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device) #to(device).long()   # (4,512)

            # accuracy
            acc = torch.sum((y == output).float() * attn)
            acc /= torch.sum(attn)
            total_acc += acc

            # calculate losses
            y_hat = y_hat.permute(0,2,1)
            loss = self.compute_loss(y_hat, y, attn)
            total_loss += loss.item()

            # udpate only in train
            if train:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        return round(total_loss/len(training_data),3), round(total_acc.item()/len(training_data),3)


    def save_checkpoint(self, epoch, train_acc, valid_acc, 
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'optimizer' : self.optim.state_dict()
        }
        torch.save(state, filename)

        best_mdl = filename.split('.')[0]+'_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)
#            torch.save(state, 'finetune-model.ckpt')

