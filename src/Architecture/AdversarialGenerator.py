import csv
from typing import Sequence, Tuple, List, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from monai.networks.nets import BasicUNet

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


from src.Architecture.Generator import Generator


class AdversarialGenerator(Generator):
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 loss: nn.Module = CrossEntropyLoss(),
                 alpha: float = 1.0):
        super().__init__(model, loss, alpha)
        self.image_shape = image_shape

        self.adversarial = torch.nn.Sequential(
            BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128)),
            nn.ReLU()
        )
        self.adversarial.eval()

        self.change = nn.Parameter(torch.zeros(*self.image_shape))

        # logging
        self.mean_changes: List[float] = []

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input = input + self.change

        # cost are the predicted change by the adversarial
        if self.alpha != 0.0:
            cost = torch.mean(torch.abs(self.adversarial(new_input)))
        else:
            cost = torch.tensor(0.0, device=input.device)

        self.mean_changes.append(self.change.mean().detach().cpu())
        return new_input, cost

    def reset(self):
        super().reset()
        device = self.change.device
        self.change = nn.Parameter(torch.zeros(*self.image_shape, device=device))

    def load_adversarial(self, name='adversarial'):
        self.adversarial.load_state_dict(
            torch.load(f"models/{name}.pth", map_location=self.change.device))

    def log_and_visualize(self,
                          image: torch.Tensor,
                          target: torch.Tensor,
                          name: str = 'generate',
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        super().log_and_visualize(image, target, name, method)

        with torch.no_grad():
            input_image, cost = self.adapt(image)
            predicted = self.adversarial(input_image).detach().cpu().numpy()
            predicted *= torch.sign(self.change).detach().cpu().numpy()

        self.save_images(name,
                         predicted_change_t1w=predicted[0, 0],
                         predicted_change_flair=predicted[0, 1],
                         bias_map_t1w=self.change[0, 0].detach().cpu(),
                         bias_map_flair=self.change[0, 1].detach().cpu(),
                         cmap='bwr',
                         norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title("predicted Change - tw1")
        plt.imshow(predicted[0, 0], cmap='bwr', norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("predicted Change - flair")
        plt.imshow(predicted[0, 1], cmap='bwr', norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title("Change - t1w")
        plt.imshow(self.change[0, 0].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title("Change - FLAIR")
        plt.imshow(self.change[0, 1].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"Results/{name}/AdversarialPrediction.png", dpi=750)
        plt.close()
        print(f"Adversarial prediction saved to Results/{name}/AdversarialPrediction.png")

    def plot_visualization(self, image: torch.Tensor, new_image: torch.Tensor):
        plt.subplot(3, 3, 1)
        plt.title("Change - t1w")
        plt.imshow(self.change[0, 0].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Change - FLAIR")
        plt.imshow(self.change[0, 1].detach().cpu(), cmap='bwr', norm=TwoSlopeNorm(0.0, vmin=-.1, vmax=.1))
        plt.axis('off')

    def plot_generation_curves(self, name: str):
        losses = np.array(self.losses)
        costs = np.array(self.costs)
        change = np.array(self.mean_changes)

        # plot loss curve
        plt.plot(losses + costs, label='complete loss')
        plt.plot(losses, label='target loss')
        plt.plot(costs, label='cost')
        plt.plot(change, label='actual image change')
        if self.alpha != 1.0:
            plt.plot(self.alpha * costs, label='alpha * cost')
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


if __name__ == '__main__':
    from src.utils import get_network, load_image, get_max_slice

    model = get_network(configuration='2d', fold=0)
    generator = AdversarialGenerator(model, (256, 256))
    generator.load_adversarial('adversarial')

    item = load_image('data/Dataset101_fcd/sub-00003')

    slice_idx, size = get_max_slice(item['target'], 2 + 1)
    print(f"selected slice: {slice_idx} with a target size of {size} pixels.")

    image = item['tensor'].select(2 + 1, slice_idx)[None]
    noise = torch.randn(image.shape) * 0.001
    image += noise
    target = item['target'].select(2 + 1, slice_idx)

    out = generator.adversarial(image).detach().cpu().numpy()
    removed = image - out * torch.sign(noise).numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("tw1")
    plt.imshow(out[0, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("flair")
    plt.imshow(out[0, 1], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("tw1")
    plt.imshow(removed[0, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("flair")
    plt.imshow(removed[0, 1], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"Adversarial.png", dpi=500)
    plt.close()
