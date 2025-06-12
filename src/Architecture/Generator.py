"""Super class for image adaption."""
import os
import warnings
from typing import Tuple, List, Literal

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch import nn
from torch.nn import CrossEntropyLoss

from src.utils import normalize, dice


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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return loss + self.alpha * cost, loss, self.alpha * cost

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Args:
            input: input tensor

        Returns:
            the adapted image and the cost of that adaption.
        """
        warnings.warn("Warning: You executed the dummy adapt function of the Generator super class!")
        return input, torch.tensor(0.0).to(input.device)

    def reset(self):
        """Resets all parameters so a new image can be generated."""
        pass

    def log_and_visualize(self,
                          image: torch.Tensor,
                          target: torch.Tensor,
                          losses: List[float],
                          target_losses: List[float],
                          costs: List[float],
                          name: str = 'generate',
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        """Visualizes the results."""
        os.makedirs(f"Results/{name}", exist_ok=True)

        loss_fn = CrossEntropyLoss()
        with torch.no_grad():
            new_image, cost = self.adapt(image)
            original_prediction = self.model(image)
            deformed_prediction = self.model(new_image)

            original_loss = loss_fn(original_prediction, target)
            deformed_loss = loss_fn(deformed_prediction, target)
            original_dice = dice(torch.argmax(original_prediction, dim=1), target)
            deformed_dice = dice(torch.argmax(deformed_prediction, dim=1), target)

            original_prediction = F.softmax(original_prediction, dim=1)[0, 1].cpu()
            deformed_prediction = F.softmax(deformed_prediction, dim=1)[0, 1].cpu()

        result = (f"Loss {original_loss} --> {deformed_loss}\n"
                  f"Adaption cost: {cost}\n"
                  f"Dice: {original_dice} --> {deformed_dice}")

        print(result)
        with open(f"Results/{name}/results.txt", "x") as f:
            f.write(result)

        self.plot_generation_curves(costs, losses, name, target_losses)

        # plot Grad Cam maps
        plt.figure(figsize=(10, 10))
        self.plot_activation_map(image, 1, method)
        self.plot_activation_map(new_image, 2, method)

        plt.tight_layout()
        plt.savefig(f"Results/{name}/GradCam.png", dpi=750)
        plt.close()
        print(f"Grad-Cam image of the results saved to Results/{name}/GradCam.png")

        # plot overview
        plt.figure(figsize=(12, 10))
        self.plot_visualization(image[0].cpu(), new_image[0].cpu())
        self.plot_original(image[0].cpu())
        self.plot_modified(new_image[0].cpu())
        self.plot_results(image[0].cpu(), target.cpu(), new_image[0].cpu(), original_prediction, deformed_prediction)

        plt.tight_layout()
        plt.savefig(f"Results/{name}/overview.png", dpi=750)
        plt.close()
        print(f"Overview of the results saved to Results/{name}/overview.png")

    def plot_generation_curves(self, costs, losses, name, target_losses):
        # plot loss curve
        plt.plot(losses, label='complete loss')
        plt.plot(target_losses, label='target loss')
        plt.plot(costs, label='cost')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Loss over generation process')
        plt.savefig(f"Results/{name}/loss_curve.png", dpi=750)
        print(f"Plot with loss and cost curves of the results saved to Results/{name}/loss_curve.png")

    def plot_visualization(self, image: torch.Tensor, new_image: torch.Tensor):
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


    def plot_activation_map(self, image, i, method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        image = image.clone()
        b, c, w, h = image.shape
        method = {'GradCAM': GradCAM, 'GradCAMPlusPlus': GradCAMPlusPlus}[method]

        with method(model=self.model, target_layers=[self.model.decoder.seg_layers[-1]]) as cam:
            activation_map = cam(input_tensor=image, targets=[SemanticSegmentationTarget(1, np.ones((w, h)))])[0, :]
            cam_image = show_cam_on_image(normalize(image[:, 0]).repeat(3, 1, 1).permute(1, 2, 0).numpy(),
                                          activation_map,
                                          use_rgb=True)

        plt.subplot(1, 2, i)
        plt.title("Modified input image" if i == 2 else "Original image")
        plt.imshow(cam_image, cmap='gray')
        plt.axis('off')
