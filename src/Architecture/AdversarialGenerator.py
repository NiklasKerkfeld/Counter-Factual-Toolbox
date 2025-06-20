from typing import Sequence, Tuple, Literal

import torch
from torch import nn

from monai.networks.nets import BasicUNet

from matplotlib import pyplot as plt

from .LossFunctions import MaskedCrossEntropyLoss
from .ChangeGenerator import ChangeGenerator

class AdversarialGenerator(ChangeGenerator):
    """Uses an Adversarial to ensure the image stays in the image domain."""
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0):
        super().__init__(model, image_shape, loss, alpha)

        self.adversarial = torch.nn.Sequential(
            BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128)),
            nn.ReLU()
        )
        self.adversarial.eval()

    def adapt(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input = image + self.change

        # cost are the predicted change by the adversarial
        if self.alpha != 0.0:
            cost = torch.mean(torch.abs(self.adversarial(new_input)))
        else:
            cost = torch.tensor(0.0, device=image.device)

        self.mean_changes.append(self.change.mean().detach().cpu())
        return new_input, cost

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
            predicted = self.adversarial(input_image)[0].detach().cpu().numpy()
            predicted *= torch.sign(self.change[0]).detach().cpu().numpy()

        change = self.change[0].detach().cpu().numpy()

        self.save_images(name,
                         predicted_change_t1w=predicted[0],
                         cmap='bwr',
                         norm=self.t1w_change_norm)

        self.save_images(name,
                         predicted_change_flair=predicted[1],
                         cmap='bwr',
                         norm=self.flair_change_norm)

        self.plot_adversarial_prediction(change, name, predicted)

    def plot_adversarial_prediction(self, change, name, predicted):
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
        plt.savefig(f"Results/{name}/AdversarialPrediction.png", dpi=750)
        plt.close()
        print(f"Adversarial prediction saved to Results/{name}/AdversarialPrediction.png")


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
