import torch
import torch.nn as nn

class TinyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(TinyBlock, self).__init__()
        
        # f1: 3x3 depthwise convolution + BatchNorm
        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # f2: 1x1 grouped pointwise convolutions with 8 groups + ReLU
        self.f2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=8, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        f1_out = self.f1(x)
        f2_out = self.f2(x + f1_out)
        out = x + f1_out + f2_out
        return out
