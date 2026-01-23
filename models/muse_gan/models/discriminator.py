import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.conv_block = nn.Sequential(
            nn.Conv3d(self.in_channels, 128, (2, 1, 1), (1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, (3, 1, 1), (1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, (1, 1, 12), (1, 1, 12)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, (1, 1, 7), (1, 1, 7)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, (1, 2, 1), (1, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, (1, 2, 1), (1, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, (1, 4, 1), (1, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, (1, 3, 1), (1, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.lin_block = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.lin_block(x)
        return x
