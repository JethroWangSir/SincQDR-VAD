import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super(Patchify, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(8, patch_size), stride=(8, patch_size), padding=0, bias=False)
        
    def forward(self, x):
        # x.shape = (batch_size, channels, height, width)
        x = self.conv(x)

        return x
