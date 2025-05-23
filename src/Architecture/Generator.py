from typing import Tuple

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, model: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.model = model
        self.alpha = alpha

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->  torch.Tensor:
        image, cost = self.image(input)
        output = self.model(image)
        loss = self.loss(output, target)
        return loss + self.alpha * cost

    def image(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return input, torch.tensor(0.0).to(input.device)