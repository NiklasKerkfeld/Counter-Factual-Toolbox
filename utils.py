import os

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from matplotlib import cm


def symetric_color_mapping(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor[0]

    # Normalize symmetrically around zero
    max_val = tensor.abs().max().item()
    normed_tensor = tensor.clamp(-max_val, max_val) / max_val  # Now in [-1, 1]

    # Convert to numpy
    normed_array = normed_tensor.cpu().numpy()

    # Use matplotlib colormap (bwr) to get RGB image
    colormap = cm.get_cmap('bwr')
    colored_image = colormap((normed_array + 1) / 2.0)  # Map [-1, 1] to [0, 1]

    # Drop alpha channel and convert to torch
    rgb_image = torch.from_numpy(colored_image[..., :3])  # shape: (H, W, 3)
    rgb_image = rgb_image.permute(2, 0, 1)  # (3, H, W)
    return rgb_image.float()

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