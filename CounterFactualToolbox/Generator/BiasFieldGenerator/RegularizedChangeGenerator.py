"""Classes for free pixel value change."""
from typing import Tuple, List, Optional

import torch
from matplotlib.colors import TwoSlopeNorm, Normalize
from torch import nn
import torch.nn.functional as F

from CounterFactualToolbox.Generator.BiasFieldGenerator.ChangeGenerator import ChangeGenerator
from CounterFactualToolbox.utils.LossFunctions import MaskedCrossEntropyLoss


class RegularizedChangeGenerator(ChangeGenerator):
    """Class for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image: torch.Tensor,
                 target: torch.Tensor,
                 name: str = 'ChangeGenerator',
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 omega: float = 1.0,
                 previous: Optional[List[torch.Tensor]] = None,
                 beta: float = 1.0,
                 width: float = 0.2
                 ):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            image: input image (B, C, H, W)
            target: target tensor (B, H, W)
            name: name to use for save files
            loss: loss function to be optimized (default: CrossEntropyLoss)
            alpha: weight of the adaption cost in comparison to the prediction loss
            omega: weight to balance between L1 regularization and smoothness regularization
            previous: List of parameters from previous runs to keep distance from
            beta: weight for the distance to previous
            width: width parameter for the Distance loss (only relevant if previous parameter
            are provided)
        """
        super().__init__(model, image, target, name, loss, alpha, previous, beta, width)
        self.omega = omega

        smoothness = torch.tensor([[[-1/8, -1/8, -1/8], [-1/8, 1, -1/8], [-1/8, -1/8, -1/8]]]).repeat(2, self.image.shape[0], 1, 1)
        self.register_buffer('smoothness', smoothness.float())

        # logging
        self.mean_changes: List[float] = []
        self.t1w_change_norm: Normalize = TwoSlopeNorm(0.0)
        self.flair_change_norm: Normalize = TwoSlopeNorm(0.0)

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        smoothness = torch.abs(F.conv2d(self.change, self.smoothness, groups=2)).mean()
        cost = torch.mean(torch.abs(self.change)) + self.omega * smoothness
        return self.image + self.change, cost
