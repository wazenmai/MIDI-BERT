import tqdm
import torch
import torch.nn as nn
import utils_bestloss as utils
from sklearn.metrics import confusion_matrix

def training(model, device, train_loader, optimizer, bs, class_num):
    model.train()
    train_loss = 0
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.float)
        #print("[train.py]",x.shape, y.shape)
        optimizer.zero_grad()
        
        y_hat = model(x) # this line could work if using gpu
        y_hat = y_hat.reshape((-1, class_num))
        y_hat = y_hat.squeeze()
        y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
        
        _, pred_label = torch.max(y_hat, dim=1)
        y_np = y.detach().cpu().numpy()
        pred_np = pred_label.detach().cpu().numpy()

        cm = confusion_matrix(y_np, pred_np, labels=[i for i in range(class_num)])
        acc, sum = 0, 0
        for i in range(1,class_num):
            acc += cm[i][i]
            for j in range(1,class_num):
                sum += cm[i][j]        
        accuracy = acc/sum
        
        CE = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        loss = CE(y_hat, y)

        loss.backward()

        train_loss += loss
        optimizer.step()
    return train_loss/len(train_loader.dataset)*bs, accuracy

def valid(model, device, valid_loader, bs, class_num):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.float)
            
            y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
            y_hat = y_hat.reshape((-1, class_num))
            y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
            
            CE = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            loss = CE(y_hat, y)
            valid_loss += loss

            _, pred_label = torch.max(y_hat, dim=1)
            y_np = y.detach().cpu().numpy()
            pred_np = pred_label.detach().cpu().numpy()

            cm = confusion_matrix(y_np, pred_np, labels=[i for i in range(class_num)])
            acc, sum = 0, 0
            for i in range(1,class_num):
                acc += cm[i][i]
                for j in range(1,class_num):
                    sum += cm[i][j]        
            accuracy = acc/sum
    return valid_loss/len(valid_loader.dataset)*bs, accuracy

