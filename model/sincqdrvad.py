import torch
import torch.nn as nn
from .sinc_conv import TimeSincExtractor, FreqSincExtractor
from .patchify import Patchify
from .csp_tiny_layer import CSPTinyLayer

class SincQDRVAD(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, patch_size, num_blocks, sinc_conv):
        super(SincQDRVAD, self).__init__()

        self.sinc_conv = sinc_conv

        if self.sinc_conv:
            self.extractor = TimeSincExtractor(out_channels=64, kernel_size=101, range_constraint=True, stride=2, bi_factor=True)
            # self.extractor = FreqSincExtractor(out_channels=64, kernel_size=101, range_constraint=True, stride=2, bi_factor=True)

        self.patchify = Patchify(in_channels, hidden_channels, patch_size)

        self.csp_tiny_layer1 = CSPTinyLayer(hidden_channels, hidden_channels, num_blocks)
        self.csp_tiny_layer2 = CSPTinyLayer(hidden_channels, hidden_channels, num_blocks)
        self.csp_tiny_layer3 = CSPTinyLayer(hidden_channels, out_channels, num_blocks)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 1),
        )

    def forward(self, x):
        if self.sinc_conv: 
            x = self.extractor(x, None)
            x = x[0]  # Untuple

        x = self.patchify(x)

        x = self.csp_tiny_layer1(x)
        x = self.csp_tiny_layer2(x)
        x = self.csp_tiny_layer3(x)

        x = self.avg_pool(x).view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def predict(self, inputs):
        logits = self.forward(inputs)
        probs = torch.sigmoid(logits)

        return probs
    
