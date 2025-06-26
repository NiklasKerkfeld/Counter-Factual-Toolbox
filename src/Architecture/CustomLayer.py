import torch
from torch import nn
from torch.nn import functional as F


class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size: int=3, sigma: float=1.0):
        super(GaussianBlurLayer, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('kernel', self.get_kernel(kernel_size, sigma))

    def get_kernel(self, kernel_size, sigma) -> torch.Tensor:
        """Returns a 2D Gaussian kernel as a torch.Tensor."""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        # Create a coordinate grid centered at 0
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')

        # Compute the 2D Gaussian
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # Normalize so the sum is 1

        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Assume input shape: (B, C, H, W)
        B, C, H, W = input.shape

        # Expand kernel to shape: (C, 1, k, k)
        kernel = self.kernel.expand(C, 1, -1, -1)

        # Apply depthwise convolution (grouped by channel)
        return F.conv2d(input, kernel, padding=self.kernel_size // 2, groups=C)