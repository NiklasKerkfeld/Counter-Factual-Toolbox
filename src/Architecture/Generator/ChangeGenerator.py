"""Classes for free pixel value change."""
import csv
from typing import Sequence, Tuple, Literal, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from torch import nn

from ..LossFunctions import MaskedCrossEntropyLoss
from .Generator import Generator


class ChangeGenerator(Generator):
    """Class for free pixel value change."""
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 beta: float = 1.0):
        """
        Classes for free pixel value change.

        Args:
            model: model used for prediction
            image_shape: shape of the input image (B, C, H, W)
            loss: loss function to be optimized (default: CrossEntropyLoss)
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, loss, alpha)
        self.image_shape = image_shape
        self.beta = beta

        self.change = nn.Parameter(torch.zeros(*image_shape))

        smoothness = torch.tensor([[[[-1/8, -1/8, -1/8], [-1/8, 1, -1/8], [-1/8, -1/8, -1/8]]]]).repeat(2, image_shape[0], 1, 1)
        self.register_buffer('smoothness', smoothness.float())

        # logging
        self.mean_changes: List[float] = []
        self.t1w_change_norm = TwoSlopeNorm(0.0)
        self.flair_change_norm = TwoSlopeNorm(0.0)

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        smoothness = nn.functional.conv2d(self.change, self.smoothness, groups=2)
        cost = torch.mean(torch.abs(self.change)) + self.beta * torch.mean(torch.abs(smoothness))
        return input + self.change, cost

    def reset(self):
        super().reset()
        device = self.change.device
        self.change = nn.Parameter(torch.zeros(*self.image_shape, device=device))

    def log_and_visualize(self,
                          image: torch.Tensor,
                          target: torch.Tensor,
                          name: str = 'generate',
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        super().log_and_visualize(image, target, name, method)

        change = self.change[0].detach().cpu()
        torch.save(change, f"Results/{name}/bias_map.pt")
        print(f"Generated bias map of image saved to Results/{name}/bias_map.pt")
        change = change.numpy()

        self.save_images(name,
                         bias_map_t1w=change[0],
                         cmap='bwr',
                         norm=self.t1w_change_norm)

        self.save_images(name,
                         bias_map_flair=change[1],
                         cmap='bwr',
                         norm=self.flair_change_norm)

    def plot_visualization(self, image: torch.Tensor, new_image: torch.Tensor):
        plt.subplot(3, 3, 1)
        plt.title("Change - t1w")
        plt.imshow(self.change[0, 0].detach().cpu(), cmap='bwr', norm=self.t1w_change_norm)
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Change - FLAIR")
        plt.imshow(self.change[0, 1].detach().cpu(), cmap='bwr', norm=self.flair_change_norm)
        plt.axis('off')

    def plot_generation_curves(self, name: str):
        losses = np.array(self.losses)
        costs = np.array(self.costs)
        change = np.array(self.mean_changes)

        # plot loss curve
        plt.plot(losses + self.alpha * costs, label='complete loss')
        plt.plot(losses, label='target loss')
        plt.plot(self.alpha * costs, label='cost')
        plt.plot(change, label='actual image change')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Loss over generation process')
        plt.savefig(f"Results/{name}/loss_curve.png", dpi=750)
        plt.close()  # Close the figure to free memory

        print(f"Plot with loss and cost curves of the results saved to Results/{name}/loss_curve.png")

        with open(f"Results/{name}/loss_curves.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'loss', 'cost', 'change'])  # Header
            writer.writerows([(i, x, y, z) for i, x, y, z in zip(np.arange(1, len(losses)+1), losses, costs, change)])

    def get_norm(self, image: torch.Tensor, new_image: torch.Tensor):
        t1w_min = min(torch.min(image[0, 0]).item(), torch.min(new_image[0, 0]).item())
        t1w_max = max(torch.max(image[0, 0]).item(), torch.max(new_image[0, 0]).item())
        t1w_extremest = max(-t1w_min, t1w_max) / 10
        self.t1w_norm = Normalize(t1w_min, t1w_max)
        self.t1w_change_norm = TwoSlopeNorm(0.0, -t1w_extremest, t1w_extremest)

        flair_min = min(torch.min(image[0, 1]).item(), torch.min(new_image[0, 1]).item())
        flair_max = max(torch.max(image[0, 1]).item(), torch.max(new_image[0, 1]).item())
        flair_extremest = max(-t1w_min, t1w_max) / 10
        self.flair_norm = Normalize(flair_min, flair_max)
        self.flair_change_norm = TwoSlopeNorm(0.0, -flair_extremest, flair_extremest)