"""Super class for image adaption."""
import os
from typing import Tuple

import numpy as np
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

    def visualize(self, image: torch.Tensor, target: torch.Tensor, name: str = 'generate'):
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

        plt.figure(figsize=(15, 12))
        self.plot_visualization(image)
        self.plot_original(image)
        self.plot_modified(new_image)
        self.plot_results(image, target, new_image, original_prediction, deformed_prediction)

        os.makedirs("Results", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"Results/{name}.png", dpi=750)
        plt.close()
        print(f"Comparison of the results saved to Results/{name}.png")

    def plot_visualization(self, image: torch.Tensor):
        plt.subplot(3, 3, 1)
        plt.title("Dummy")
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Dummy")
        plt.axis('off')

    def plot_original(self, image):
        plt.subplot(3, 3, 2)
        plt.title("Original - t1w")
        plt.imshow(image[0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 5)
        plt.title("Original - FLAIR")
        plt.imshow(image[1], cmap='gray')
        plt.axis('off')

    def plot_modified(self, new_image):
        plt.subplot(3, 3, 3)
        plt.title("Modified - t1w")
        plt.imshow(new_image[0], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.title("Modified - FLAIR")
        plt.imshow(new_image[1], cmap='gray')
        plt.axis('off')

    def plot_results(self, image, target, new_image, original_prediction, deformed_prediction):
        plt.subplot(3, 3, 7)
        plt.title("Target")
        plt.imshow(image[0], cmap='gray')
        plt.imshow(
            np.concatenate((target, np.zeros_like(target), np.zeros_like(target), target > .1),
                           axis=0).astype(float).transpose(1, 2, 0), alpha=0.3)
        plt.axis('off')

        plt.subplot(3, 3, 8)
        plt.title("Original prediction")
        plt.imshow(image[0], cmap='gray')
        plt.imshow(np.concatenate(
            (original_prediction[None], np.zeros_like(original_prediction[None]),
             np.zeros_like(original_prediction[None]), original_prediction[None] > .1),
            axis=0).astype(float).transpose(1, 2, 0), alpha=0.3)
        plt.axis('off')

        plt.subplot(3, 3, 9)
        plt.title("Modified prediction")
        plt.imshow(new_image[0], cmap='gray')
        plt.imshow(np.concatenate(
            (deformed_prediction[None], np.zeros_like(deformed_prediction[None]),
             np.zeros_like(deformed_prediction[None]), deformed_prediction[None] > .1),
            axis=0).astype(float).transpose(1, 2, 0), alpha=0.3)
        plt.axis('off')
