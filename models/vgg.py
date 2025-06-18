import math
import torch
import torch.nn as nn

from models.hefnet import polynom_act
from models.channel_attn import SEBlock, CBAM

class VGG5(nn.Module):
    def __init__(self,in_channels=3, input_size=32, num_classes=2, kernel_size=3, dropout=0.0, attn=None):
        super(VGG5, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout

        activation = polynom_act() #nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=1, bias=False)
        self.relu1 = activation
        self.avgpool1 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel_size,padding=(self.kernel_size-1)//2, stride=1, groups=8, bias=False)
        self.relu2 = activation
        self.avgpool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=1, groups=8, bias=False)
        self.relu3 =activation
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=1, groups=8, bias=False)
        self.relu4 = activation
        self.avgpool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        if attn is None:
            self.attn = None
        elif attn == 'SE':
            self.attn = SEBlock(256)
        elif attn == 'CBAM':
            self.attn = CBAM(256, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=True)
        
        d = input_size // 16
        self.linear1 = nn.Linear(256*d*d, num_classes, bias=False)

        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_() 
        #     elif isinstance(m, nn.Linear):
        #         n = m.weight.size(1)
        #         m.weight.data.normal_(0, 0.01)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        out = self.relu2(self.conv3(out))
        out = self.avgpool3(out)
        out = self.relu4(self.conv4(out))
        if self.attn is not None:
            out = self.attn(out)
        out = self.avgpool4(out)

        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        
        return out