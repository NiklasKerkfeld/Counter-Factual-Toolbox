import glob
from typing import Optional

import torch
from torch.optim import Adam
from tqdm import trange

from src.Architecture.Generator import Generator
from src.utils import load_image, get_max_slice


def generate(path: str, generator: Generator, optimizer: Adam, steps: int = 100, slice_idx: Optional[int] = None,
             slice_dim: int = 2, name: Optional[str] = None):
    name = name if name is not None else generator.__class__.__name__

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    print(f"Using device: {device}")

    item = load_image(path)

    if slice_idx is None:
        slice_idx, size = get_max_slice(item['target'], slice_dim + 1)
        print(f"selected slice: {slice_idx} with a target size of {size} pixels.")

    image = item['tensor'].select(slice_dim + 1, slice_idx)[None].to(device)
    target = item['target'].select(slice_dim + 1, slice_idx).to(device)

    print("starting process...")
    bar = trange(steps, desc='generating...')

    for _ in bar:
        optimizer.zero_grad()
        loss = generator(image, target)
        loss.backward()
        optimizer.step()

        bar.set_description(f"loss: {loss.detach().cpu().item()}")

    print("\nstarting logging...\n")
    generator.log_and_visualize(image,
                                target,
                                f"{len(glob.glob(f'Results/*'))}_{name}", 'GradCAM')