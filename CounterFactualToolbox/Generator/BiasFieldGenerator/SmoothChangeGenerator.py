"""Classes for free pixel value change."""
from typing import Tuple, List, Optional

import torch
from matplotlib.colors import TwoSlopeNorm, Normalize
from torch import nn

from CounterFactualToolbox.Generator.BiasFieldGenerator.ChangeGenerator import ChangeGenerator
from CounterFactualToolbox.utils.CustomLayer import GaussianBlurLayer
from CounterFactualToolbox.utils.LossFunctions import MaskedCrossEntropyLoss


class SmoothChangeGenerator(ChangeGenerator):
    """Class for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image: torch.Tensor,
                 target: torch.Tensor,
                 name: str = 'SmoothChangeGenerator',
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 kernel_size: int = 3,
                 sigma: float = 1.0,
                 previous: Optional[List[torch.Tensor]] = None,
                 beta: float = 1.0,
                 width: float = 0.2):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            image: image tensor (B, C, W, H)
            target: target tensor (B, W, H)
            name: name to use for save files
            loss: loss function to optimize
            alpha: weight of the adaption cost in comparison to the prediction loss
            previous: List of parameters from previous runs to keep distance from (B, C, W, H)
            beta: weight for the distance to previous
            width: width parameter for the Distance loss (only relevant if previous parameter
            are provided)
        """
        super().__init__(model, image, target, name, loss, alpha, previous, beta, width)
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

    def _modify(self, param):
        return self.smooth(param)
