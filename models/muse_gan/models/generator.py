import torch
import torch.nn as nn


class TemporalGenerator(nn.Module):
    def __init__(self, latent_dim=32, output_dim=32):
        super(TemporalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim, out_channels=1024, kernel_size=2, stride=2
            ),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                in_channels=1024, out_channels=output_dim, kernel_size=3, stride=1
            ),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):  # # B x 32
        x = x.unsqueeze(2)  # B x 32 x 1
        x = self.block(x)
        return x


class BarGenerator(nn.Module):
    def __init__(self, input_size=128, out_channels=1):
        super(BarGenerator, self).__init__()  # B x 5 x 1 x 1
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=1024,
                kernel_size=(2, 1),
                stride=(2, 1),
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=(2, 1), stride=(2, 1)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=(2, 1), stride=(2, 1)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=(2, 1), stride=(2, 1)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=(3, 1), stride=(3, 1)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=(1, 7), stride=(1, 7)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=(1, 12),
                stride=(1, 12),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.block(x)
