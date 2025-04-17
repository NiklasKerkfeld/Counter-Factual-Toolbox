import os

import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.model(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(out_channels // 4, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvLayer(in_channels, out_channels, kernel_size=3, padding=1) if in_channels != out_channels else None

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        if self.conv4 is not None:
            x = self.conv4(x)

        x = h + x
        x = self.pool(x)

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 2):
        super().__init__()
        self.block1 = Block(in_channels, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, 512)

        self.upsample1 = Upsample(512, 256)
        self.upsample2 = Upsample(512, 128)
        self.upsample3 = Upsample(256, 64)
        self.upsample4 = Upsample(128, 64)

        self.head = nn.Conv2d(64, classes, kernel_size=1, padding=0)

    def forward(self, x):
        h1 = self.block1(x)  # 64
        h2 = self.block2(h1)  # 128
        h3 = self.block3(h2)  # 256
        h4 = self.block4(h3)  # 512

        u1 = self.upsample1(h4)  # 256
        u2 = self.upsample2(torch.cat((u1, h3), dim=1))  # 256 + 256 = 512 --> 128
        u3 = self.upsample3(torch.cat((u2, h2), dim=1))  # 128 + 128 = 256 --> 64
        u4 = self.upsample4(torch.cat((u3, h1), dim=1))  # 64 + 64 = 128 --> 64

        return self.head(u4)

    def save(self, name='SimpleUNet'):
        os.makedirs("saved_models", exist_ok=True)
        torch.save(self.state_dict(), f"saved_models/{name}.pt")

    def load(self, name='SimpleUNet'):
        self.load_state_dict(torch.load(f"saved_models/{name}.pt"))


if __name__ == '__main__':
    dummy = torch.randn(1, 3, 64, 64)

    model = SimpleUNet()

    print(model(dummy).shape)
