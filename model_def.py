import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256]):
        super(UNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            in_channels = feature

        for feature in reversed(features):
            self.decoders.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx, decoder in enumerate(self.decoders):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = nn.functional.pad(x, [0, skip_connection.shape[3]-x.shape[3], 0, skip_connection.shape[2]-x.shape[2]])
            x = torch.cat((skip_connection, x), dim=1)
            x = decoder(x)
        return torch.clamp(self.final_conv(x), 0.0, 1.0)
