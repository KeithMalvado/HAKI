import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dummy = nn.Identity()

    def forward(self, x):
        return x
