import os

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

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
        h1 = self.block1(x)     # 64
        h2 = self.block2(h1)    # 128
        h3 = self.block3(h2)    # 256
        h4 = self.block4(h3)    # 512

        u1 = self.upsample1(h4)                                    # 256
        u2 = self.upsample2(torch.cat((u1, h3), dim=1))     # 256 + 256 = 512 --> 128
        u3 = self.upsample3(torch.cat((u2, h2), dim=1))     # 128 + 128 = 256 --> 64
        u4 = self.upsample4(torch.cat((u3, h1), dim=1))     # 64 + 64 = 128 --> 64

        return self.head(u4)

    def predict(self, image):
        with torch.no_grad():
            pred = torch.nn.functional.softmax(self(image), dim=1)

        return pred[:, 0]

    def save(self):
        os.makedirs("saved_models", exist_ok=True)
        torch.save(self.state_dict(), "saved_models/SimpleUNet.pt")

    def load(self):
        self.load_state_dict(torch.load("saved_models/SimpleUNet.pt"))


if __name__ == '__main__':
    dummy = torch.randn(1, 3, 64, 64)

    model = SimpleUNet()

    print(model(dummy).shape)
