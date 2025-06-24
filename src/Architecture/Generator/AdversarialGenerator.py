from typing import Sequence, Tuple, Literal

import torch
from torch import nn

from monai.networks.nets import BasicUNet

from matplotlib import pyplot as plt

from src.Architecture.CustomLayer import GaussianBlurLayer
from ..LossFunctions import MaskedCrossEntropyLoss
from .ChangeGenerator import ChangeGenerator


class AdversarialGenerator(ChangeGenerator):
    """Uses an Adversarial to ensure the image stays in the image domain."""

    def __init__(self, model: nn.Module,
                 image: torch.tensor,
                 target: torch.tensor,
                 name: str = "AdversarialGenerator",
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0):
        super().__init__(model, image, target, name, loss, alpha)

        self.adversarial = torch.nn.Sequential(
            BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128)),
            nn.ReLU()
        )
        self.adversarial.eval()

        with torch.no_grad():
            init_pred: torch.Tensor = self.adversarial(image)
            self.register_buffer('init_pred', init_pred.float())

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input = self.image + self.change

        # cost are the predicted change by the adversarial
        if self.alpha != 0.0:
            pred = self.adversarial(new_input) - self.init_pred
            cost = torch.mean(torch.abs(pred))
        else:
            cost = torch.tensor(0.0, device=self.image.device)

        self.mean_changes.append(torch.abs(self.parameter).mean().detach().cpu())
        return new_input, cost

    def load_adversarial(self, name='adversarial'):
        self.adversarial.load_state_dict(
            torch.load(f"models/{name}.pth", map_location=self.parameter.device))

    def log_and_visualize(self,
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        super().log_and_visualize(method)

        input_image, cost = self.adapt()
        predicted = self.adversarial(input_image)[0].detach().cpu().numpy()
        predicted *= torch.sign(self.parameter[0]).detach().cpu().numpy()

        change = self.parameter[0].detach().cpu().numpy()

        self.save_images(predicted_change_t1w=predicted[0],
                         cmap='bwr',
                         norm=self.t1w_change_norm)

        self.save_images(predicted_change_flair=predicted[1],
                         cmap='bwr',
                         norm=self.flair_change_norm)

        self.plot_adversarial_prediction(change, predicted)

    def plot_adversarial_prediction(self, change, predicted):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title("predicted Change - tw1")
        plt.imshow(predicted[0], cmap='bwr', norm=self.t1w_change_norm)
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.title("predicted Change - flair")
        plt.imshow(predicted[1], cmap='bwr', norm=self.flair_change_norm)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.title("Change - t1w")
        plt.imshow(change[0], cmap='bwr', norm=self.t1w_change_norm)
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.title("Change - FLAIR")
        plt.imshow(change[1], cmap='bwr', norm=self.flair_change_norm)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"Results/{self.name}/AdversarialPrediction.png", dpi=750)
        plt.close()
        print(f"Adversarial prediction saved to Results/{self.name}/AdversarialPrediction.png")
