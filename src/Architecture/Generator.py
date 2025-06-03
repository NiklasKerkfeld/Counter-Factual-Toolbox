"""Super class for image adaption."""

from typing import Tuple

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import CrossEntropyLoss


class Generator(nn.Module):
    """Super class for image adaption."""

    def __init__(self, model: nn.Module, loss=CrossEntropyLoss(), alpha: float = 1.0):
        """
        Super class for image adaption.

        Args:
            model: model used for prediction
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__()
        self.model = model
        self.alpha = alpha

        self.loss = loss

        self.model.eval()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward path for generation.

        Adopts image with adopt function. Then calculates prediction with loss and add them with the
        cost of the adoption.

        Args:
            input: input tensor
            target: target tensor

        Returns:
            over all loss
        """
        image, cost = self.adapt(input)
        output = self.model(image)
        loss = self.loss(output, target)
        return loss + self.alpha * cost

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Args:
            input: input tensor

        Returns:
            the adapted image and the cost of that adaption.
        """
        return input, torch.tensor(0.0).to(input.device)

    def reset(self):
        """Resets all parameters so a new image can be generated."""
        pass

    def visualize(self, image: torch.Tensor, target: torch.Tensor):
        """Visualizes the results."""
        with torch.no_grad():
            new_image, _ = self.adapt(image)
            original_prediction = self.model(image)
            deformed_prediction = self.model(new_image)

            original_prediction = F.softmax(original_prediction, dim=1)[0, 1].cpu()
            deformed_prediction = F.softmax(deformed_prediction, dim=1)[0, 1].cpu()

        image = image[0, 0].cpu()
        new_image = new_image[0, 0].cpu()
        target = target.cpu()

        # Original image - 2 channels stacked vertically
        plt.subplot(3, 3, 2)
        plt.title("Original - t1w")
        plt.imshow(image[..., 0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 5)
        plt.title("Original - FLAIR")
        plt.imshow(image[..., 1], cmap='gray')
        plt.axis('off')

        # Modified image - 2 channels stacked vertically
        plt.subplot(3, 3, 3)
        plt.title("Modified - t1w")
        plt.imshow(new_image[..., 0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.title("Modified - FLAIR")
        plt.imshow(new_image[..., 1], cmap='gray')
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
