import csv
from typing import Sequence, List, Tuple, Literal

import numpy as np
from scipy.stats import pearsonr

import torch
from torch import nn

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from .AdversarialGenerator import AdversarialGenerator
from ..LossFunctions import MaskedCrossEntropyLoss, DistanceLoss


class DifferenceAdversarialGenerator(AdversarialGenerator):
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 change_list: List[torch.Tensor],
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 beta: float = 1.0):
        super().__init__(model, image_shape, loss, alpha)

        # self.change = nn.Parameter(torch.randn(*image_shape) * 0.01)
        self.register_buffer('change_list', torch.stack(change_list).float())
        self.beta = beta
        self.distance_loss = DistanceLoss(width=.2)
        self.distance_loss_list = []

    def adapt(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input = image + self.parameter

        # cost are the predicted change by the adversarial
        if self.alpha != 0.0:
            cost = torch.mean(torch.abs(self.adversarial(new_input)))
        else:
            cost = torch.tensor(0.0, device=image.device)

        # cost are the predicted change by the adversarial
        if self.beta != 0.0:
            dist_loss = self.distance_cost()
            cost += self.beta * dist_loss
            self.distance_loss_list.append(dist_loss)

        self.mean_changes.append(torch.abs(self.parameter).mean().detach().cpu())
        return new_input, cost

    def distance_cost(self):
        return torch.sum(torch.tensor([self.distance_loss(self.parameter, change) for change in self.change_list]))

    def log_and_visualize(self,
                          image: torch.Tensor,
                          target: torch.Tensor,
                          name: str = 'generate',
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        super().log_and_visualize(image, target, name, method)

        change = self.parameter[0].detach().cpu().numpy()
        result = ""
        for i, alt_change in enumerate(self.change_list):
            alt_change = alt_change.detach().cpu().numpy()
            corr_result = pearsonr(change.flatten(), alt_change.flatten())
            result += f"distance to image {i}: {np.sum(np.abs(change - alt_change))}\n"
            result += f"pearsonr correlation to image {i}: statistic={corr_result.statistic}, p value={corr_result.pvalue}\n"

        print(result)
        with open(f"Results/{name}/results.txt", "a") as f:
            f.write(f"\n{result}")

        self.plot_different_bias_maps(change, image[0].detach().cpu().numpy(), name)

    def plot_different_bias_maps(self, change, image, name):
        n = self.change_list.shape[0] + 1

        plt.figure(figsize=(10, 10))

        for i in range(n):
            map = change if i == 0 else self.change_list[i-1].detach().cpu().numpy()

            j = i + 1
            plt.subplot(4, n, j)
            plt.title("Change - tw1")
            plt.imshow(map[0], cmap='bwr', norm=self.t1w_change_norm)
            plt.axis('off')
            plt.subplot(4, n, j+n)
            plt.title("Change - flair")
            plt.imshow(map[1], cmap='bwr', norm=self.flair_change_norm)
            plt.axis('off')
            plt.subplot(4, n, j+2*n)
            plt.title("New - t1w")
            plt.imshow(image[0] + map[0], cmap='gray', norm=Normalize())
            plt.axis('off')
            plt.subplot(4, n, j+3*n)
            plt.title("New - FLAIR")
            plt.imshow(image[1] + map[1], cmap='gray', norm=Normalize())
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"Results/{name}/MultipleBiasMaps.png", dpi=750)
        plt.close()
        print(f"Overview over the Bias maps saved to Results/{name}/MultipleBiasMaps.png")

    def plot_generation_curves(self, name: str):
        losses = np.array(self.losses)
        costs = np.array(self.costs)
        change = np.array(self.mean_changes)
        distance = np.array(self.distance_loss_list)

        # plot loss curve
        plt.plot(losses + costs, label='complete loss')
        plt.plot(losses, label='target loss')
        plt.plot(costs, label='cost')
        plt.plot(change, label='actual image change')
        if self.alpha != 1.0:
            plt.plot(self.alpha * costs, label='alpha * cost')
        plt.plot(distance, '--', label='distance')
        if self.beta != 1.0:
            plt.plot(self.beta * distance, '--', label='beta * distance')

        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Loss over generation process')
        plt.savefig(f"Results/{name}/loss_curve.png", dpi=750)
        plt.close()  # Close the figure to free memory

        print(f"Plot with loss and cost curves of the results saved to Results/{name}/loss_curve.png")

        with open(f"Results/{name}/loss_curves.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'loss', 'cost', 'change', 'distance'])  # Header
            writer.writerows([(i, w, x, y, z) for i, w, x, y, z in zip(np.arange(1, len(losses)+1), losses, costs, change, distance)])