"""Classes for free pixel value change."""
from typing import Tuple, List

import torch
from matplotlib.colors import TwoSlopeNorm, Normalize
from torch import nn
import torch.nn.functional as F

from .ChangeGenerator import ChangeGenerator
from ..LossFunctions import MaskedCrossEntropyLoss


class RegularizedChangeGenerator(ChangeGenerator):
    """Class for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image: torch.Tensor,
                 target: torch.Tensor,
                 name: str = 'ChangeGenerator',
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 beta: float = 1.0):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            image_shape: shape of the input image (B, C, H, W)
            loss: loss function to be optimized (default: CrossEntropyLoss)
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, image, target, name, loss, alpha)
        self.beta = beta

        smoothness = torch.tensor([[[-1/8, -1/8, -1/8], [-1/8, 1, -1/8], [-1/8, -1/8, -1/8]]]).repeat(2, self.image.shape[0], 1, 1)
        self.register_buffer('smoothness', smoothness.float())

        # logging
        self.mean_changes: List[float] = []
        self.t1w_change_norm: Normalize = TwoSlopeNorm(0.0)
        self.flair_change_norm: Normalize = TwoSlopeNorm(0.0)

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        smoothness = torch.abs(F.conv2d(self.change, self.smoothness, groups=2)).mean()
        cost = torch.mean(torch.abs(self.change)) + self.beta * smoothness
        return self.image + self.change, cost
