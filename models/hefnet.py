import math
import torch
import torch.nn as nn

from models.channel_attn import SEBlock, CBAM

class polynom_act(nn.Module):
    def __init__(self):
        super(polynom_act, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.c = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

    def forward(self, x):
        return (self.alpha * (x ** 2) + self.beta * x + self.c)


class hefNet(nn.Module):
    def __init__(self, in_channels=3, input_size=32, num_classes=2, channel_list=[64,128,256], attn=None):
        super(hefNet, self).__init__()
        activation = polynom_act()

        self.conv1 = nn.Conv2d(in_channels, channel_list[0], kernel_size=3, padding=0, stride=1)
        self.bn1=nn.BatchNorm2d(channel_list[0])
        self.relu1 = activation

        self.conv2 = nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, stride=1, groups=8)
        self.bn2=nn.BatchNorm2d(channel_list[1])
        self.relu2 = activation
        self.avgpool1=nn.AvgPool2d(kernel_size=2,stride=2)
        
        self.conv3 = nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, stride=1, groups=8)
        self.bn3=nn.BatchNorm2d(channel_list[2])
        self.relu3 = activation
        self.avgpool2=nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv4 = nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, stride=1, groups=8)
        if input_size == 16:
            self.conv4 = nn.Conv2d(channel_list[2], channel_list[2], kernel_size=1, stride=1, groups=8)
        self.bn4=nn.BatchNorm2d(channel_list[2])
        self.relu4 = activation
        self.avgpool3=nn.AvgPool2d(kernel_size=2,stride=2)

        if attn is None:
            self.attn = None
        elif attn == 'SE':
            self.attn = SEBlock(channel_list[2])
        elif attn == 'CBAM':
            self.attn = CBAM(channel_list[2], reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=True)
        
        d = (((input_size-4) // 2 -2) // 2 - 2 )// 2
        if input_size == 16: d = 1

        self.linear1 = nn.Linear(channel_list[2]*d*d, num_classes)

        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        
        x = self.avgpool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.avgpool2(self.relu3(self.bn3(self.conv3(x))))
        x = self.bn4(self.conv4(x))
        if self.attn is not None:
            x = self.attn(x)
        x = self.avgpool3(self.relu4(x))
        
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
   
        return x
