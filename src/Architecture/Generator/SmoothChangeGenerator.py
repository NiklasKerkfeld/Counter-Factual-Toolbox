"""Classes for free pixel value change."""
from typing import Tuple, List

import torch
from matplotlib.colors import TwoSlopeNorm, Normalize
from torch import nn
import torch.nn.functional as F

from . import ChangeGenerator
from ..CustomLayer import GaussianBlurLayer
from ..LossFunctions import MaskedCrossEntropyLoss


class SmoothChangeGenerator(ChangeGenerator):
    """Class for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image: torch.Tensor,
                 target: torch.Tensor,
                 name: str = 'SmoothChangeGenerator',
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 kernel_size: int = 3,
                 sigma: float = 1.0):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            loss: loss function to be optimized (default: CrossEntropyLoss)
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, image, target, name, loss, alpha)
        self.smooth = GaussianBlurLayer(kernel_size=kernel_size, sigma=sigma)

        # logging
        self.mean_changes: List[float] = []
        self.t1w_change_norm: Normalize = TwoSlopeNorm(0.0)
        self.flair_change_norm: Normalize = TwoSlopeNorm(0.0)

    @property
    def change(self):
        return self.smooth(self.parameter)

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cost = torch.mean(torch.abs(self.change))
        return self.image + self.change, cost

    def reset(self):
        super().reset()
        self.parameter = nn.Parameter(torch.zeros(*self.image_shape, device=self.parameter.device))
