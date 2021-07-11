import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SelfAttention(nn.Module):
    """SelfAttention class"""
    def __init__(self, input_dim: int, da: int, r: int) -> None:
        """Instantiating SelfAttention class
        Args:
            input_dim (int): dimension of input, eg) (batch_size, seq_len, input_dim)
            da (int): the number of features in hidden layer from self-attention
            r (int): the number of aspects of self-attention
        """
        super(SelfAttention, self).__init__()
        self._ws1 = nn.Linear(input_dim, da, bias=False)
        self._ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_mat = F.softmax(self._ws2(torch.tanh(self._ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self._projection = (in_channels != out_channels)
        self._ops = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv1d(out_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU())

        if self._projection:
            self._shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, 1),
                                           nn.BatchNorm1d(out_channels))
        self._bn = nn.BatchNorm1d(out_channels)
        self._activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self._shortcut(x) if self._projection else x
        fmap = self._ops(x) + shortcut
        fmap = self._activation(self._bn(fmap))
        return fmap

class Conv1d_mp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1):
        super(Conv1d_mp, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._ops = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self._activation = nn.ReLU()
        self._mp = nn.MaxPool1d(2,2)

    def forward(self, x):
        return self._mp(self._activation(self._ops(x)))

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(BiLSTM, self).__init__()
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, hc = self._ops(x)
        feature = torch.cat([*hc[0]], dim=1)
        return feature

class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvolutionLayer, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels // 3, stride=1, kernel_size=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels // 3, stride=4, kernel_size=4
        )
        self.conv3 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels // 3, stride=8, kernel_size=8
        )

    def forward(self, x):
        conv1_fmap = F.relu(self.conv1(x))
        conv2_fmap = F.relu(self.conv2(x))
        conv3_fmap = F.relu(self.conv3(x))
        return conv1_fmap, conv2_fmap, conv3_fmap

class MaxOverTimePooling(nn.Module):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        conv1_fmap, conv2_fmap, conv3_fmap = x
        fmap = torch.cat(
            [
                conv1_fmap.max(dim=-1)[0],
                conv2_fmap.max(dim=-1)[0],
                conv3_fmap.max(dim=-1)[0],
            ],
            dim=-1,
        )
        return fmap