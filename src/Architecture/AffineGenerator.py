from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .LossFunctions import MaskedCrossEntropyLoss
from .Generator import Generator


class AffineGenerator(Generator):
    def __init__(self, model: nn.Module, loss=MaskedCrossEntropyLoss(), alpha: float = 1.0):
        super(AffineGenerator, self).__init__(model, loss, alpha)

        self.change = nn.Parameter(torch.zeros(2, 2, 3), requires_grad=True)
        self.register_buffer('base', torch.Tensor([[1, 0, 0], [0, 1, 0]]))

    @property
    def theta(self):
        return self.base + self.change

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = F.affine_grid(self.theta, [2, 1, *input.shape[2:]])
        new_image = F.grid_sample(input.permute(1, 0, 2, 3), grid, padding_mode='reflection').permute(1, 0, 2, 3)
        return new_image, torch.sum(torch.abs(self.change))
