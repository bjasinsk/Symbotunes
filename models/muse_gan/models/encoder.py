import torch.nn as nn
import torch.nn.functional as F


class BarEncoder(nn.Module):
    def __init__(self, input_channels):
        super(BarEncoder, self).__init__()
        self.input_channels = input_channels
        self.block = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=16,
                    kernel_size=(1, 12),
                    stride=(1, 12),
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=(1, 7), stride=(1, 7)
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=(3, 1), stride=(3, 1)
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 1)
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 1)
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 1)
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=(2, 1), stride=(2, 1)
                ),
            ]
        )

    def forward(self, x):
        encodings = []
        for layer in self.block:
            x = F.relu(F.batch_norm(layer(x)))
            encodings.append(x)
        return encodings
