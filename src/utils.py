import numpy as np
import matplotlib.pyplot as plt
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def visualize_deformation_field(image, dx, dy, scale=1, color='red'):
    """
    Visualize a 2D elastic deformation vector field on an image.

    Parameters:
        image (2D or 3D array): The base image (grayscale or RGB).
        dx (2D array): Displacement in x-direction.
        dy (2D array): Displacement in y-direction.
        scale (float): Arrow scaling factor.
        color (str): Color of the arrows.
    """
    image_height, image_width = image.shape
    height, width = dx.shape

    X, Y = np.meshgrid(np.arange(0, image_width, image_width // width),
                       np.arange(0, image_height, image_height // height))

    plt.figure(figsize=(20, 20))
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    plt.quiver(X, Y, dx, dy, color=color, angles='xy', scale_units='xy', scale=1 / scale)

    plt.title(f'Elastic Deformation Field (scale x{scale})')
    plt.axis('off')
    plt.savefig('logs/deformation.png', dpi=1000)
    plt.close()


def inverse_z_transform(image: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    image *= std
    image += mean
    return image.clip(0.0, None)


def normalize(image: torch.Tensor) -> torch.Tensor:
    image -= image.min()
    image /= image.max()
    return image


def get_network(configuration: str, fold: int = 0):
    predictor = nnUNetPredictor()

    predictor.initialize_from_trained_model_folder(
        f"models/Dataset101_fcd/nnUNetTrainer__nnUNetPlans__{configuration}",
        str(fold),
        "checkpoint_best.pth")

    net = predictor.network

    return net


def dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten()
    target = target.flatten()

    intersection = torch.sum(torch.logical_and(pred, target))

    return (2 * intersection) / (torch.sum(pred) + torch.sum(target))
