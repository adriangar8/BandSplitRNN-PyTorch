import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # squeeze: average over time
        self.fc1 = nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channel, time]
        y = self.avg_pool(x)  # [batch, channel, 1]
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y
