from typing import Optional

import torch


def intersection_over_union(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.flatten()
    target = target.flatten()

    intersection = torch.sum(torch.logical_and(pred, target))
    union = torch.sum(torch.logical_or(pred, target))

    if intersection == 0 and union == 0:
        return 1.0

    if union == 0:
        return 0.0

    return intersection.item() / union.item()


def normalize(image: torch.Tensor, shift: Optional[float] = None, scale: Optional[float] = None) -> torch.Tensor:
    shift = image.min() if shift is None else shift
    image -= shift

    scale = image.max() if scale is None else scale
    image /= scale
    return image


if __name__ == '__main__':
    print(intersection_over_union(torch.zeros((10, 10, 10)), torch.zeros((10, 10, 10))))
