"""Classes for free pixel value change."""

from typing import Sequence, Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch import nn
import torch.nn.functional as F
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


    def visualize(self, image: torch.Tensor, target: torch.Tensor):
        """Visualizes the results."""
        with torch.no_grad():
            new_image, _ = self.adapt(image)
            original_prediction = self.model(image)
            deformed_prediction = self.model(new_image)

            original_prediction = F.softmax(original_prediction, dim=1)[0, 1].cpu()
            deformed_prediction = F.softmax(deformed_prediction, dim=1)[0, 1].cpu()

        image = image[0].cpu()
        new_image = new_image[0].cpu()
        target = target.cpu()

        # Plotting
        plt.subplot(3, 3, 1)
        plt.title("Change - t1w")
        plt.imshow(self.change[0, 0].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0))
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Change - FLAIR")
        plt.imshow(self.change[0, 1].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0))
        plt.axis('off')

        # Original image - 2 channels stacked vertically
        plt.subplot(3, 3, 2)
        plt.title("Original - t1w")
        plt.imshow(image[0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 5)
        plt.title("Original - FLAIR")
        plt.imshow(image[1], cmap='gray')
        plt.axis('off')

        # Modified image - 2 channels stacked vertically
        plt.subplot(3, 3, 3)
        plt.title("Modified - t1w")
        plt.imshow(new_image[0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.title("Modified - FLAIR")
        plt.imshow(new_image[1], cmap='gray')
        plt.axis('off')

        # Remaining images
        plt.subplot(3, 3, 7)
        plt.title("Target")
        plt.imshow(target[0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 8)
        plt.title("Original prediction")
        plt.imshow(original_prediction, cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 9)
        plt.title("Modified prediction")
        plt.imshow(deformed_prediction, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("logs/result.png", dpi=750)
        plt.close()

        print(f"Comparison of the results saved to logs/result.png")
