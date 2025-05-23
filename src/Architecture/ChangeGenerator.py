from typing import Sequence, Tuple

import torch
from torch import nn

from src.Architecture.Generator import Generator


class ChangeGenerator(Generator):
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 alpha: float = 1.0):
        super().__init__(model, alpha)
        self.image_shape = image_shape

        self.change = nn.Parameter(torch.zeros(*image_shape))

    def image(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return input + self.change, torch.mean(self.change)
    

