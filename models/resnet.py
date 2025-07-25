import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

from models.hefnet import polynom_act
from models.channel_attn import SEBlock, CBAM

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, attn=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = polynom_act()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        if attn is None:
            self.attn = None
        elif attn == 'SE':
            self.attn = SEBlock(planes)
        elif attn == 'CBAM':
            self.attn = CBAM(planes, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attn is not None:
            out = self.attn(out)
        out = out + self.shortcut(x)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[3,3,3], attn=None, input_size=32, 
                 in_channels=4, num_channels=16, in_planes=16, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.layer1 = self._make_layer(block, attn, num_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, attn, num_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, attn, num_channels*4, num_blocks[2], stride=2)
        self.linear = nn.Linear(num_channels*4, num_classes)

        self.relu = polynom_act()

    def _make_layer(self, block, attn, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, attn))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out