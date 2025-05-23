from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.Architecture.Generator import Generator


class ElasticDeformation2D(Generator):
    def __init__(self, model: nn.Module,
                 image_shape: Tuple[int, int],
                 parameter_grid_shape: Tuple[int, int],
                 alpha: float = 1.0):
        super().__init__(model, alpha)

        self.H, self.W = self.image_shape = image_shape
        self.grid_shape = parameter_grid_shape

        self.dx = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))
        self.dy = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))

        self.x, self.y = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        self.x = self.x.float()
        self.y = self.y.float()

    def grid(self):
        nx = F.interpolate(self.dx, self.image_shape, mode='bilinear')
        ny = F.interpolate(self.dy, self.image_shape, mode='bilinear')

        # Create meshgrid
        x = self.x + nx
        y = self.y + ny

        # Normalize coordinates to [-1, 1]
        x = 2.0 * x / (self.H - 1) - 1.0
        y = 2.0 * y / (self.W - 1) - 1.0

        return torch.stack((y, x), dim=-1).squeeze(0)  # Shape: (1, H, W, D, 2)

    def image(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        grid = self.grid()
        new_image = F.grid_sample(input, grid, padding_mode='reflection', align_corners=True)
        return new_image, torch.mean(self.dx) + torch.mean(self.dy)


class ElasticDeformation3D(Generator):
    def __init__(self, model: nn.Module,
                 image_shape: Tuple[int, int, int],
                 parameter_grid_shape: Tuple[int, int, int],
                 alpha: float = 1.0):
        super().__init__(model, alpha)

        self.H, self.W, self.D = self.image_shape = image_shape
        self.grid_shape = parameter_grid_shape

        self.dx = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))
        self.dy = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))
        self.dz = nn.Parameter(torch.zeros(1, 1, *self.grid_shape))

        self.x, self.y, self.z = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), torch.arange(self.D), indexing='ij')
        self.x = self.x.float()
        self.y = self.y.float()
        self.z = self.z.float()


    def grid(self):
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

    def image(self, input: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = self.grid()
        new_image = F.grid_sample(input, grid, padding_mode='reflection', align_corners=True)
        return new_image, torch.mean(self.dx) + torch.mean(self.dy) + torch.mean(self.dz)

