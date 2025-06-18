import numpy as np
import torch
import torch.nn as nn

from models.hefnet import polynom_act
from models.channel_attn import SEBlock, CBAM

class LeNet5(nn.Module):
    def __init__(self, in_channels=3, input_size=32, num_classes=2, attn=None):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.relu1 = polynom_act()
        self.avgpool1=nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1,groups=4)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.relu2= polynom_act()
        self.avgpool2=nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1)

        if attn is None:
            self.attn = None
        elif attn == 'SE':
            self.attn = SEBlock(128)
        elif attn == 'CBAM':
            self.attn = CBAM(128, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=True)

        nn.init.xavier_uniform_(self.conv3.weight)
        self.relu3= polynom_act()
        self.avgpool3=nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(128,64)
        self.relu4= polynom_act()
        self.linear2 = nn.Linear(64,num_classes)

        
    def forward(self, x):
        out = self.avgpool1(self.relu1(self.conv1(x)))
        out = self.avgpool2(self.relu2(self.conv2(out)))
        out = self.conv3(out)
        if self.attn is not None:
            out = self.attn(out)
        out = self.avgpool3(self.relu3(out))

        out = out.reshape(out.shape[0],-1)        
        out = self.relu4(self.linear1(out))
        out = self.linear2(out)

        return out