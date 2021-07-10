# -*- coding: utf-8 -*-
"""
Created on Dec 14 2020

@author: Yi-Hui (Sophia) Chou
"""

import numpy as np
from torch.utils.data import Dataset
import torch

class PopDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """
    def __init__(self, X, y):
        self.data = X 
        self.label = y

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
