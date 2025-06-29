from typing import Optional

import torch


def intersection_over_union(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten()
    target = target.flatten()

    intersection = torch.sum(torch.logical_and(pred, target))

    return (2 * intersection) / (torch.sum(pred) + torch.sum(target) + 1e-6)


def normalize(image: torch.Tensor, shift: Optional[float] = None, scale: Optional[float] = None) -> torch.Tensor:
    shift = image.min() if shift is None else shift
    image -= shift

    scale = image.max() if scale is None else scale
    image /= scale
    return image