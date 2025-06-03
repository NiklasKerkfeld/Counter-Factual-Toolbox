"""Classes for free pixel value change."""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from src.Architecture.Generator import Generator


class ChangeGenerator(Generator):
    """Classes for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 alpha: float = 1.0):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            image_shape: shape of the input image
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, alpha)
        self.image_shape = image_shape

        self.change = nn.Parameter(torch.zeros(*image_shape))

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return input + self.change, torch.mean(torch.abs(self.change))

