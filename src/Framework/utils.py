from io import BytesIO
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def normalize(image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    image -= image.min()
    image /= image.max()
    return image


def change_visualization(change: torch.Tensor, background: Optional[torch.Tensor] = None, figsize=(16, 16)) -> torch.Tensor:
    change = change.cpu().numpy()
    abs_change = np.abs(change).sum(0)

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    if background is not None:
        ax.imshow(background, cmap='gray')
    cax = ax.imshow(abs_change, cmap='Reds', vmin=abs_change.min(), vmax=abs_change.max(), alpha=.5)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, alpha=1.0)
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