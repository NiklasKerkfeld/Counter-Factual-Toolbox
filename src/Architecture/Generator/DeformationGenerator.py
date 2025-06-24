"""Classes for 2D and 3D Deformation"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from ..LossFunctions import MaskedCrossEntropyLoss
from .Generator import Generator


class ElasticDeformation(Generator):
    """"2D elastic deformation."""
    def __init__(self, model: nn.Module,
                 image: torch.Tensor,
                 target: torch.Tensor,
                 name: str = "ElasticDeformation",
                 parameter_grid_shape: Tuple[int, int] = (20, 32),
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 0.1):
        """
        2D elastic deformation.

        Args:
            model: model used for prediction
            image_shape: shape of the input image (B, C, H, W)
            parameter_grid_shape: number of parameters in every image dimension
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, image, target, name, loss, alpha)

        self.B, self.C, self.H, self.W = self.image_shape = self.image.shape
        self.grid_shape = parameter_grid_shape

        self.dx = nn.Parameter(torch.zeros(self.B, 1, *self.grid_shape))
        self.dy = nn.Parameter(torch.zeros(self.B, 1, *self.grid_shape))

        x, y = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        self.register_buffer('x', x.repeat(self.B, 1, 1).float())
        self.register_buffer('y', y.repeat(self.B, 1, 1).float())

    def grid(self) -> torch.Tensor:
        """
        Interpolates the grid for every pixel from the parameter grid.

        Returns:
            grid with deformation for every pixel
        """
        nx = F.interpolate(self.dx, (self.H, self.W), mode='bilinear')
        ny = F.interpolate(self.dy, (self.H, self.W), mode='bilinear')

        # Create meshgrid
        x = self.x + nx
        y = self.y + ny

        # Normalize coordinates to [-1, 1]
        x = 2.0 * x / (self.H - 1) - 1.0
        y = 2.0 * y / (self.W - 1) - 1.0

        return torch.stack((y, x), dim=-1).squeeze(0)  # Shape: (1, H, W, 2)

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the given image via elastic deformation.

        Args:
            input: input image

        Returns:
            adapted image and the cost of that adaption
        """
        grid = self.grid()
        new_image = F.grid_sample(self.image, grid, padding_mode='reflection')
        return new_image, torch.mean(torch.abs(self.dx)) + torch.mean(torch.abs(self.dy))

    def plot_visualization(self, new_image: torch.Tensor):
        image_height, image_width = self.image[0, 0].shape
        height, width = self.dx[0, 0].shape

        X, Y = np.meshgrid(np.arange(0, image_width, image_width // width),
                           np.arange(0, image_height, image_height // height))

        plt.subplot(3, 3, 1)
        plt.title("Deformation - original")
        plt.imshow(self.image[0, 0].detach().cpu(), cmap='gray')
        dx = self.dx[0, 0].detach().cpu().numpy()
        dy = self.dy[0, 0].detach().cpu().numpy()
        plt.quiver(X+dy, Y+dx, -dy, -dx, color='red',
                   angles='xy', scale_units='xy', scale=1)
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Deformation - deformed")
        plt.imshow(new_image[0], cmap='gray')
        dx = self.dx[0, 0].detach().cpu().numpy()
        dy = self.dy[0, 0].detach().cpu().numpy()
        plt.quiver(X+dy, Y+dx, -dy, -dx, color='red',
                   angles='xy', scale_units='xy', scale=1)
        plt.axis('off')
