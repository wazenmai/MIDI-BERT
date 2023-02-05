import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

from MidiBERT.finetune_model import TokenClassification, SequenceClassification


class FinetuneTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, test_dataloader, layer, 
                lr, class_num, hs, testset_shape, cpu, cuda_devices=None, model=None, SeqClass=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:',self.device)
        self.midibert = midibert
        self.SeqClass = SeqClass
        self.layer = layer

        if model != None:    # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model, sequence-level task?', SeqClass)
            if SeqClass:
                self.model = SequenceClassification(self.midibert, class_num, hs).to(self.device)
            else:
                self.model = TokenClassification(self.midibert, class_num, hs).to(self.device)

#        for name, param in self.model.named_parameters():
#            if 'midibert.bert' in name:
#                    param.requires_grad = False
#            print(name, param.requires_grad)


        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader
        
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.testset_shape = testset_shape
    
    def compute_loss(self, predict, target, loss_mask, seq):
        loss = self.loss_func(predict, target)
        if not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss)/loss.shape[0]
        return loss

 
    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, 0, self.SeqClass)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        valid_loss, valid_acc = self.iteration(self.valid_data, 1, self.SeqClass)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        test_loss, test_acc, all_output = self.iteration(self.test_data, 2, self.SeqClass)
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode, seq):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_cnt, total_loss = 0, 0, 0

        if mode == 2: # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]
            x, y = x.to(self.device), y.to(self.device)     # seq: (batch, 512, 4), (batch) / token: , (batch, 512)

            # avoid attend to pad word
            if not seq:
                attn = (y != 0).float().to(self.device)   # (batch,512)
            else:   
                attn = torch.ones((batch, 512)).to(self.device)     # attend each of them

            y_hat = self.model.forward(x, attn, self.layer)     # seq: (batch, class_num) / token: (batch, 512, class_num)

            # get the most likely choice with max
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                all_output[cnt : cnt+batch] = output
                cnt += batch

            # accuracy
            if not seq:
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0,2,1)
            loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss.item()

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        if mode == 2:
            return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4), all_output
        return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4)


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

