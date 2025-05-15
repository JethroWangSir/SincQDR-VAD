import torch
import torch.nn as nn
from .tiny_block import TinyBlock

class CSPTinyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPTinyLayer, self).__init__()
        
        # Split channels
        self.split_channels = in_channels // 2

        # TinyBlocks
        self.tiny_blocks = nn.Sequential(
            *[TinyBlock(self.split_channels, self.split_channels) for _ in range(num_blocks)]
        )
        
        # Transition layer to adjust channel dimensions
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Split input into two parts
        p1 = x[:, :self.split_channels, :, :]
        p2 = x[:, self.split_channels:, :, :]

        # Process p2 through TinyBlocks
        p2_out = self.tiny_blocks(p2)
        
        # Concatenate p1 and processed p2
        concatenated = torch.cat((p1, p2_out), dim=1)
        
        # Apply transition layer
        out = self.transition(concatenated)
        return out
