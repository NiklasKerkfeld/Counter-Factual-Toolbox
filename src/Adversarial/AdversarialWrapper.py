"""Wrapper class to train Adversarial."""

from typing import Tuple, Optional

import torch
from torch import nn


class AdversarialWrapper(nn.Module):
    def __init__(self,
                 segmentation: nn.Module,
                 adversarial: nn.Module,
                 input_shape: Tuple[int, int, int, int]):
        super(AdversarialWrapper, self).__init__()
        self.generator = segmentation
        self.adversarial = adversarial
        self.input_shape = input_shape
        self.mode = 'adversarial'

        print(f"{self.input_shape=}")

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

        self.change = nn.Parameter(torch.zeros(self.input_shape))

    def get_input(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.change

    def generate_mode(self):
        self.mode = 'generate'
        self.adversarial.eval()
        for param in self.adversarial.parameters():
            param.requires_grad = False

    def train_mode(self):
        self.mode = 'adversarial'
        self.adversarial.train()
        for param in self.adversarial.parameters():
            param.requires_grad = True

    def reset(self):
        with torch.no_grad():
            self.change.data.zero_()

    def forward(self, x) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.mode == 'generate':
            new_image = self.get_input(x)
            segmentation = self.generator(new_image)
            adversarial = self.adversarial(new_image)

            return segmentation, adversarial

        else:
            return None, self.adversarial(x)