"""Classes for 2D and 3D Deformation"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from src.Architecture.Generator import Generator
from src.utils import visualize_deformation_field


class ElasticDeformation2D(Generator):
    """"2D elastic deformation."""
    def __init__(self, model: nn.Module,
                 image_shape: Tuple[int, int],
                 parameter_grid_shape: Tuple[int, int],
                 loss: nn.Module = CrossEntropyLoss(),
                 alpha: float = 1.0):
        """
        2D elastic deformation.

        Args:
            model: model used for prediction
            image_shape: shape of the input image
            parameter_grid_shape: number of parameters in every image dimension
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, loss, alpha)

        self.H, self.W = self.image_shape = image_shape
        self.grid_shape = parameter_grid_shape

        self.dx = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))
        self.dy = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))

        x, y = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        self.register_buffer('x', x.float())
        self.register_buffer('y', y.float())

    def grid(self) -> torch.Tensor:
        """
        Interpolates the grid for every pixel from the parameter grid.

        Returns:
            grid with deformation for every pixel
        """
        nx = F.interpolate(self.dx, self.image_shape, mode='bilinear')
        ny = F.interpolate(self.dy, self.image_shape, mode='bilinear')

        # Create meshgrid
        x = self.x + nx
        y = self.y + ny

        # Normalize coordinates to [-1, 1]
        x = 2.0 * x / (self.H - 1) - 1.0
        y = 2.0 * y / (self.W - 1) - 1.0

        return torch.stack((y, x), dim=-1).squeeze(0)  # Shape: (1, H, W, 2)

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the given image via elastic deformation.

        Args:
            input: input image

        Returns:
            adapted image and the cost of that adaption
        """
        grid = self.grid()
        new_image = F.grid_sample(input, grid, padding_mode='reflection')
        return new_image, torch.mean(torch.abs(self.dx)) + torch.mean(torch.abs(self.dy))

    def visualize(self, image: torch.Tensor, target: torch.Tensor):
        super().visualize(image, target)

        with torch.no_grad():
            new_image, _ = self.adapt(image)
            new_image = new_image[0, 0].cpu()

        visualize_deformation_field(new_image,
                                    self.dx[0, 0].detach().cpu().numpy(),
                                    self.dy[0, 0].detach().cpu().numpy(),
                                    scale=1)



class ElasticDeformation3D(Generator):
    """3D elastic deformation."""
    def __init__(self, model: nn.Module,
                 image_shape: Tuple[int, int, int],
                 parameter_grid_shape: Tuple[int, int, int],
                 loss: nn.Module = CrossEntropyLoss(),
                 alpha: float = 1.0):
        super().__init__(model, loss, alpha)
        """
        3D elastic deformation.
        
        Args:
            model: model used for prediction
            image_shape: shape of the input image
            parameter_grid_shape: number of parameters in every image dimension
            alpha: weight of the adaption cost in comparison to the prediction loss
        """

        self.H, self.W, self.D = self.image_shape = image_shape
        self.grid_shape = parameter_grid_shape

        self.dx = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))
        self.dy = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))
        self.dz = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))

        x, y, z = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), torch.arange(self.D), indexing='ij')
        self.register_buffer('x', x.float())
        self.register_buffer('y', y.float())
        self.register_buffer('z', z.float())

    def grid(self) -> torch.Tensor:
        """
        Interpolates the grid for every pixel from the parameter grid.

        Returns:
            grid with deformation for every voxel
        """
        nx = F.interpolate(self.dx, self.image_shape, mode='trilinear')
        ny = F.interpolate(self.dy, self.image_shape, mode='trilinear')
        nz = F.interpolate(self.dz, self.image_shape, mode='trilinear')

        # Create meshgrid
        x = self.x + nx
        y = self.y + ny
        z = self.z + nz

        # Normalize coordinates to [-1, 1]
        x = 2.0 * x / (self.H - 1) - 1.0
        y = 2.0 * y / (self.W - 1) - 1.0
        z = 2.0 * z / (self.D - 1) - 1.0

        return torch.stack((y, x, z), dim=-1).squeeze(0)  # Shape: (1, H, W, D, 3)

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the given image via elastic deformation.

        Args:
            input: input image

        Returns:
            adapted image and the cost of that adaption
        """
        grid = self.grid()
        new_image = F.grid_sample(input, grid, padding_mode='reflection', align_corners=True)
        cost = torch.mean(torch.abs(self.dx)) + torch.mean(torch.abs(self.dy)) + torch.mean(torch.abs(self.dz))
        return new_image, cost

