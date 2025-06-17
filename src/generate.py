import glob
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import trange

from src.Architecture.CustomLoss import MaskedCrossentropy
from src.Architecture.Generator import Generator
from src.utils import get_network, load_image, get_max_slice, dice


def main(path: str, generator: Generator, optimizer: Adam, steps: int = 100, slice_idx: Optional[int] = None,
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


if __name__ == '__main__':
    from src.Architecture import (ChangeGenerator,
                                  ElasticDeformation2D,
                                  AffineGenerator,
                                  AdversarialGenerator,
                                  ScaleAndShiftGenerator,
                                  AffineGenerator)

    model = get_network(configuration='2d', fold=0)
    loss = MaskedCrossentropy()

    # generator = ScaleAndShiftGenerator(model, (1, 2, 160, 256), loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.scale, generator.shift], lr=1e-1)

    # generator = AffineGenerator(model, loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.change], lr=1e-1)

    # generator = ElasticDeformation2D(model, (1, 2, 160, 256), (20, 32), loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.dx, generator.dy], lr=1e-1)

    # generator = ChangeGenerator(model, (1, 2, 160, 256), loss=loss, alpha=1.0)
    # optimizer = torch.optim.Adam([generator.change], lr=1e-1)

    generator = AdversarialGenerator(model, (1, 2, 160, 256), loss=loss, alpha=1.0)
    generator.load_adversarial()
    optimizer = torch.optim.Adam([generator.change], lr=1e-3)

    # interesting: sub-00003, sub-00043, sub-00048, sub-00116
    main('data/Dataset101_fcd/sub-00003', generator, optimizer)
