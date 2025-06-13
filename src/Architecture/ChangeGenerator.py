"""Classes for free pixel value change."""
from typing import Sequence, Tuple, Literal

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch import nn
from torch.nn import CrossEntropyLoss

from src.Architecture.Generator import Generator


class ChangeGenerator(Generator):
    """Classes for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 loss: nn.Module = CrossEntropyLoss(),
                 alpha: float = 1.0):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            image_shape: shape of the input image (B, C, H, W)
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, loss, alpha)
        self.image_shape = image_shape

        self.change = nn.Parameter(torch.zeros(*image_shape))

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return input + self.change, torch.mean(torch.abs(self.change))

    def log_and_visualize(self,
                          image: torch.Tensor,
                          target: torch.Tensor,
                          name: str = 'generate',
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        super().log_and_visualize(image, target, name, method)

        self.save_images(name,
                         bias_map_t1w=self.change[0, 0].detach().cpu(),
                         bias_map_flair=self.change[0, 1].detach().cpu(),
                         cmap='bwr',
                         norm=TwoSlopeNorm(0.0))

    def plot_visualization(self, image: torch.Tensor, new_image: torch.Tensor):
        plt.subplot(3, 3, 1)
        plt.title("Change - t1w")
        plt.imshow(self.change[0, 0].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0))
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Change - FLAIR")
        plt.imshow(self.change[0, 1].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0))
        plt.axis('off')
