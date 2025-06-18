import torch
import torch.nn as nn
import torch.nn.functional as F

class polynom_act(nn.Module):
    def __init__(self):
        super(polynom_act, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.c = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

    def forward(self, x):
        return (self.alpha * (x ** 2) + self.beta * x + self.c)


class approxSigmoid(nn.Module):
    def __init__(self):
        super(approxSigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([-0.00001]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([0.12]), requires_grad=True)
        self.c = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def forward(self, x):
        return  self.alpha * x ** 3 + self.beta * x  + self.c
    
def approxSigmoid_(x):
    return -0.0010833 * x ** 3 + 0.12 * x + 0.5
    
# Squeeze and Excitation module
class SEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SEBlock, self).__init__()
        if planes % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(planes, planes // reduction, bias=True)
        self.lelu = polynom_act()
        self.linear2 = nn.Linear(planes // reduction, planes, bias=True)
        self.sigmoid = approxSigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.lelu(self.linear1(y))
        y = self.sigmoid(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y
    
    
# Convolutional Block Attention Module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv.weight.data.zero_()
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = polynom_act() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class ApproxAdaptiveMaxPool2d(nn.Module):
    def __init__(self):
        super(ApproxAdaptiveMaxPool2d, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([0.001]), requires_grad=True)
    
    def forward(self, x):
        return self.alpha * (x * x).mean(dim=(-1, -2), keepdim=True)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            polynom_act(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = ApproxAdaptiveMaxPool2d()
        self.sigmoid = approxSigmoid()

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


class approx_max(nn.Module):
    def __init__(self, dim=1):
        super(approx_max, self).__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.FloatTensor([0.001]), requires_grad=True)

    def forward(self, x):
        return self.alpha *  (x * x).mean(dim=self.dim, keepdim=True)


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()
        self.max_ = approx_max(dim=1)

    def forward(self, x):
        return torch.cat((self.max_(x), torch.mean(x,1).unsqueeze(1)), dim=1) #torch.mean(x,1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = approxSigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting

        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        return x_out