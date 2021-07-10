from torch.utils.data import Dataset
import torch

class MidiDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """
    def __init__(self, X):
        self.data = X 

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index])
