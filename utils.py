import os

import matplotlib as mpl
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def symetric_color_mapping(tensor: torch.Tensor, figsize=(16, 16)) -> torch.Tensor:
    """
    Converts a (N, N) tensor to a (3, H, W) image tensor with a colorbar using bwr colormap.
    Args:
        tensor (torch.Tensor): 2D tensor of shape (N, N)
        figsize (tuple): Size of the matplotlib figure
    Returns:
        torch.Tensor: Image tensor with shape (3, H, W), values in [0, 1]
    """
    array = tensor[0].cpu().numpy()

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(array, cmap='bwr', vmin=-1, vmax=1)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    ax.axis('off')

    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)

    # Load image from buffer and convert to tensor
    image = Image.open(buf).convert('RGB')
    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    return img_tensor


def plot(image, mask, change, pred_before, pred_after, loss_curve):
    centered_norm = mpl.colors.CenteredNorm()
    # norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=False)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 0.05, 1, 0.05])

    axs = [[None for _ in range(5)] for _ in range(3)]

    axs[0][0] = fig.add_subplot(gs[0, 0])
    axs[0][0].set_title("image")
    axs[0][0].imshow(image[0], cmap='gray')
    axs[0][0].axis('off')

    axs[0][1] = fig.add_subplot(gs[0, 1])
    axs[0][1].set_title("mask")
    axs[0][1].imshow(mask, cmap='gray')
    axs[0][1].axis('off')

    axs[0][3] = fig.add_subplot(gs[0, 3])
    axs[0][3].set_title("pred before")
    axs[0][3].imshow(pred_before, cmap='gray')
    axs[0][3].axis('off')

    axs[1][0] = fig.add_subplot(gs[1, 0])
    axs[1][0].set_title("changed image")
    axs[1][0].imshow(image[0] + change[0], cmap='gray')
    axs[1][0].axis('off')

    axs[1][1] = fig.add_subplot(gs[1, 1])
    axs[1][1].set_title("change")
    im_change = axs[1][1].imshow(change[0], norm=centered_norm, cmap='bwr')
    axs[1][1].axis('off')
    fig.colorbar(im_change, cax=fig.add_subplot(gs[1, 2]))

    axs[1][3] = fig.add_subplot(gs[1, 3])
    axs[1][3].set_title("pred after")
    axs[1][3].imshow(pred_after, cmap='gray')
    axs[1][3].axis('off')

    # Create one large subplot spanning (2, 0) and (2, 1)
    ax_loss = fig.add_subplot(gs[2, 0:3])
    ax_loss.set_title("Loss Curve")
    ax_loss.plot(loss_curve, color='blue')
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")

    axs[2][3] = fig.add_subplot(gs[2, 3])
    axs[2][3].set_title("difference pred")
    im_diff = axs[2][3].imshow(pred_after - pred_before, norm=centered_norm, cmap='bwr')
    axs[2][3].axis('off')
    fig.colorbar(im_diff, ax=axs[2][3])

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/result.png")
