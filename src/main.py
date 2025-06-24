import glob
from typing import Optional

import torch
from torch.optim import Adam

from src.Architecture.Generator import Generator
from src.utils import get_image


def generate(image: torch.Tensor,
             target: torch.Tensor,
             generator: Generator,
             optimizer: Adam,
             steps: int = 100,
             name: Optional[str] = None):
    name = name if name is not None else generator.__class__.__name__
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator.to(device)
    image = image.to(device)
    target = target.to(device)

    generator.generate(image, optimizer, steps, target)

    print("\nstarting logging...\n")
    generator.log_and_visualize(image,
                                target,
                                f"{len(glob.glob(f'Results/*'))}_{name}", 'GradCAM')
