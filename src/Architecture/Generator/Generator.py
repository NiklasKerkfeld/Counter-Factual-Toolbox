"""Super class for image adaption."""
import csv
import os
import warnings
from typing import Tuple, List, Literal, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch.nn import CrossEntropyLoss
from tqdm import trange

from ..LossFunctions import MaskedCrossEntropyLoss
from src.utils import normalize, intersection_over_union


class Generator(nn.Module):
    """Super class for image adaption."""

    def __init__(self,
                 model: nn.Module,
                 image: torch.Tensor,
                 target: torch.Tensor,
                 name: str = 'Generator',
                 loss=MaskedCrossEntropyLoss(),
                 alpha: float = 1.0):
        """
        Super class for image adaption.

        Args:
            model: model used for prediction
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__()
        self.model = model
        self.register_buffer('image', image)
        self.register_buffer('target', target)
        self.loss = loss
        self.alpha = alpha

        self.model.eval()

        # for logging
        self.losses: List[float] = []
        self.costs: List[float] = []

        self.t1w_norm = Normalize()
        self.flair_norm = Normalize()

    def generate(self, optimizer, steps):
        print("starting process...")
        bar = trange(steps, desc='generating...')
        for _ in bar:
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            optimizer.step()

            bar.set_description(f"loss: {loss.detach().cpu().item()}")

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward path for generation.

        Adopts image with adopt function. Then calculates prediction with loss and add them with the
        cost of the adoption.

        Returns:
            over all loss
        """
        image, cost = self.adapt()
        output = self.model(image)
        loss = self.loss(output, self.target)

        self.losses.append(loss.detach().cpu())
        self.costs.append(cost.detach().cpu())

        return loss + self.alpha * cost

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Returns:
            the adapted image and the cost of that adaption.
        """
        warnings.warn("Warning: You executed the dummy adapt function of the Generator super class!")
        return self.imagew, torch.tensor(0.0).to(self.image.device)

    def reset(self):
        """Resets all parameters so a new image can be generated."""
        self.losses = []
        self.costs = []

    def log_and_visualize(self,
                          method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        """Visualizes the results."""
        os.makedirs(f"Results/{self.name}", exist_ok=True)

        new_image, original_prediction, deformed_prediction = self.generate_results()

        # get norms to ensure same normalization in images
        self.get_norm(new_image)

        # plot Grad Cam maps
        self.generate_gradcam(method, new_image)

        # move everything to cpu and numpy
        image = self.image[0].detach().cpu().numpy()
        new_image = new_image[0].detach().cpu().numpy()
        original_prediction = original_prediction.detach().cpu().numpy()
        deformed_prediction = deformed_prediction.detach().cpu().numpy()
        target = self.target.detach().cpu().numpy()

        self.plot_generation_curves()

        # save images
        self.save_images(original_t1w=image[0], adapted_t1w=new_image[0], norm=self.t1w_norm)
        self.save_images(original_flair=image[1], adapted_flair=new_image[1], norm=self.flair_norm)
        self.save_images(original_target=target[0], original_prediction=original_prediction, adapted_prediction=deformed_prediction)

        # plot overview
        self.plot_overview(image, target, new_image, original_prediction, deformed_prediction)
        self.create_gif_of_adaption(image[0], new_image[0], self.t1w_norm, 't1w')
        self.create_gif_of_adaption(image[1], new_image[1], self.flair_norm, 'flair')

    def generate_results(self):
        loss_fn = CrossEntropyLoss()
        with torch.no_grad():
            new_image, cost = self.adapt()
            original_prediction = self.model(self.image)
            deformed_prediction = self.model(new_image)

            original_loss = loss_fn(original_prediction, self.target)
            deformed_loss = loss_fn(deformed_prediction, self.target)
            original_iou = intersection_over_union(torch.argmax(original_prediction, dim=1), self.target)
            deformed_iou = intersection_over_union(torch.argmax(deformed_prediction, dim=1), self.target)

            original_prediction = F.softmax(original_prediction, dim=1)[0, 1].cpu()
            deformed_prediction = F.softmax(deformed_prediction, dim=1)[0, 1].cpu()
        result = (f"Loss {original_loss} --> {deformed_loss}\n"
                  f"Adaption cost: {cost}\n"
                  f"IoU: {original_iou} --> {deformed_iou}")
        print(result)
        with open(f"Results/{self.name}/results.txt", "x") as f:
            f.write(result)
        return new_image, original_prediction, deformed_prediction

    def get_norm(self, new_image):
        t1w_min = min(torch.min(self.image[0, 0]).item(), torch.min(new_image[0, 0]).item())
        t1w_max = max(torch.max(self.image[0, 0]).item(), torch.max(new_image[0, 0]).item())
        self.t1w_norm = Normalize(t1w_min, t1w_max)

        flair_min = min(torch.min(self.image[0, 1]).item(), torch.min(new_image[0, 1]).item())
        flair_max = max(torch.max(self.image[0, 1]).item(), torch.max(new_image[0, 1]).item())
        self.flair_norm = Normalize(flair_min, flair_max)


    def plot_overview(self,
                      image: torch.Tensor,
                      target: torch.Tensor,
                      new_image: torch.Tensor,
                      original_prediction: torch.Tensor,
                      deformed_prediction: torch.Tensor,
                      ):
        plt.figure(figsize=(12, 10))
        self.plot_visualization(new_image)
        self.plot_original(image,)
        self.plot_modified(new_image)
        self.plot_results(image, target, new_image, original_prediction, deformed_prediction)
        plt.tight_layout()
        plt.savefig(f"Results/{self.name}/overview.png", dpi=750)
        plt.close()
        print(f"Overview of the results saved to Results/{self.name}/overview.png")

    def plot_visualization(self, new_image: torch.Tensor):
        plt.subplot(3, 3, 1)
        plt.title("Dummy")
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.title("Dummy")
        plt.axis('off')

    def plot_original(self, image: torch.Tensor):
        plt.subplot(3, 3, 2)
        plt.title("Original - t1w")
        plt.imshow(image[0], cmap='gray', norm=self.t1w_norm)
        plt.axis('off')

        plt.subplot(3, 3, 5)
        plt.title("Original - FLAIR")
        plt.imshow(image[1], cmap='gray', norm=self.flair_norm)
        plt.axis('off')

    def plot_modified(self, new_image: torch.Tensor):
        plt.subplot(3, 3, 3)
        plt.title("Modified - t1w")
        plt.imshow(new_image[0], cmap='gray', norm=self.t1w_norm)
        plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.title("Modified - FLAIR")
        plt.imshow(new_image[1], cmap='gray', norm=self.flair_norm)
        plt.axis('off')

    def plot_results(self, image, target, new_image, original_prediction, deformed_prediction):
        plt.subplot(3, 3, 7)
        plt.title("Target")
        plt.imshow(image[0], cmap='gray', norm=self.t1w_norm)
        plt.imshow(
            np.concatenate((target, np.zeros_like(target), np.zeros_like(target), target > .1),
                           axis=0).astype(float).transpose(1, 2, 0), alpha=0.3)
        plt.axis('off')

        plt.subplot(3, 3, 8)
        plt.title("Original prediction")
        plt.imshow(image[0], cmap='gray', norm=self.t1w_norm)
        plt.imshow(np.concatenate(
            (original_prediction[None], np.zeros_like(original_prediction[None]),
             np.zeros_like(original_prediction[None]), original_prediction[None] > .1),
            axis=0).astype(float).transpose(1, 2, 0), alpha=0.3)
        plt.axis('off')

        plt.subplot(3, 3, 9)
        plt.title("Modified prediction")
        plt.imshow(new_image[0], cmap='gray', norm=self.t1w_norm)
        plt.imshow(np.concatenate(
            (deformed_prediction[None], np.zeros_like(deformed_prediction[None]),
             np.zeros_like(deformed_prediction[None]), deformed_prediction[None] > .1),
            axis=0).astype(float).transpose(1, 2, 0), alpha=0.3)
        plt.axis('off')

    def generate_gradcam(self, method, new_image: torch.Tensor):
        plt.figure(figsize=(10, 10))
        self.plot_activation_map(self.image, 1, method)
        self.plot_activation_map(new_image, 2, method)
        plt.tight_layout()
        plt.savefig(f"Results/{self.name}/GradCam.png", dpi=750)
        plt.close()
        print(f"Grad-Cam image of the results saved to Results/{self.name}/GradCam.png")

    def plot_activation_map(self, image, i, method: Literal['GradCAM', 'GradCAMPlusPlus'] = 'GradCAM'):
        image = image.clone()
        b, c, w, h = image.shape
        method = {'GradCAM': GradCAM, 'GradCAMPlusPlus': GradCAMPlusPlus}[method]

        with method(model=self.model, target_layers=[self.model.decoder.seg_layers[-1]]) as cam:
            activation_map = cam(input_tensor=image, targets=[SemanticSegmentationTarget(1, np.ones((w, h)))])[0, :]
            cam_image = show_cam_on_image(normalize(image[:, 0]).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
                                          activation_map,
                                          use_rgb=True)

        plt.subplot(1, 2, i)
        plt.title("Modified input image" if i == 2 else "Original image")
        plt.imshow(cam_image, cmap='gray')
        plt.axis('off')

    def save_images(self, cmap: str = 'gray', norm: Optional[Normalize] = None, **kwargs: torch.Tensor):
        for key, image in kwargs.items():
            for i in range(2):
                plt.figure()  # Ensure a new figure is created for each image
                im = plt.imshow(image, cmap=cmap, norm=norm)
                plt.axis('off')
                if i == 1:
                    plt.colorbar(im, fraction=0.046, pad=0.04)  # Add colorbar
                plt.tight_layout()
                plt.savefig(f"Results/{self.name}/{key}{'_colorbar' if i == 1 else ''}.png", bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the figure to free memory
                print(f"{key} saved to Results/{self.name}/{key}{'_colorbar' if i == 1 else ''}.png")

    def plot_generation_curves(self):
        losses = np.array(self.losses)
        costs = np.array(self.costs)

        # plot loss curve
        plt.plot(losses + costs, label='complete loss')
        plt.plot(losses, label='target loss')
        plt.plot(costs, label='cost')
        if self.alpha != 1.0:
            plt.plot(self.alpha * costs, label='alpha * cost')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Loss over generation process')
        plt.savefig(f"Results/{self.name}/loss_curve.png", dpi=750)
        plt.close()  # Close the figure to free memory

        print(f"Plot with loss and cost curves of the results saved to Results/{self.name}/loss_curve.png")

        with open(f"Results/{self.name}/loss_curves.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'loss', 'cost'])  # Header
            writer.writerows([(i, x, y) for i, x, y in zip(np.arange(1, len(losses)+1), losses, costs)])



    def create_gif_of_adaption(self, original_image: torch.Tensor, adapted_image: torch.Tensor, norm: Normalize, sequence: str):
        """
        Creates an animated GIF from two numpy array images.

        Parameters:
        - original_image: original image
        - adapted_image: adapted image
        - name: Name of the generation run.
        - sequence: Name of the sequence.
        """
        original_image = Image.fromarray(np.uint8(normalize(original_image, norm.vmin, norm.vmax)  * 255))
        adapted_image = Image.fromarray(np.uint8(normalize(adapted_image, norm.vmin, norm.vmax) * 255))

        # Save as GIF
        original_image.save(
            f"Results/{self.name}/modification_{sequence}.gif",
            save_all=True,
            append_images=[adapted_image],
            duration=500,
            loop=0  # loop forever
        )
        print(f"Gif of {sequence} adaption saved to Results/{self.name}/modification_{sequence}.gif")

