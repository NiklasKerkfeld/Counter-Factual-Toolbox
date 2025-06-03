import glob
import os
from copy import deepcopy
from typing import Dict, Optional

import monai
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import SaveImage, Compose, LoadImaged, ResampleToMatchd, NormalizeIntensityd, \
    DivisiblePadd, ConcatItemsd, ToTensord, CastToTypeD, ToDeviced
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from src.Architecture.CustomTransforms import AddMissingd, SelectSliced


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

    print(f"Visualization of the deformation saved to logs/deformation.png")


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


def save(image: torch.Tensor, path: str, example: monai.data.MetaTensor, post_fix:str = 'pred', dtype = np.int8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image = SaveImage(output_dir=path,
                           output_postfix=post_fix,
                           output_dtype=dtype,
                           separate_folder=False)
    output_image = deepcopy(example)
    output_image.set_array(image)

    save_image(output_image)


def get_vram(device: torch.device):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free = free_bytes / 1024 ** 3  # Convert to GB
    total = total_bytes / 1024 ** 3  # Convert to GB
    used = total - free
    return used, total, free


def get_image_files(path: str):
    name = os.path.basename(path)
    item = {
        't1w': glob.glob(f"{path}/anat/{name}*T1w.nii.gz")[0],
        'FLAIR': glob.glob(f"{path}/anat/{name}*FLAIR.nii.gz")[0]
    }
    roi_paths = list(glob.glob(f"{path}/anat/{name}*FLAIR_roi.nii.gz"))
    if len(roi_paths) > 0:
        item['target'] = roi_paths[0]

    return item


def load_item(item: Dict[str, str],
              device: Optional[torch.device] = None,
              slice: Optional[int] = None):
    device = device if device is not None else 'cpu'

    loader = Compose([
        LoadImaged(keys=['t1w', 'FLAIR', 'target'],
                   reader="NibabelReader",
                   ensure_channel_first=True,
                   allow_missing_keys=True),
        AddMissingd(keys=['t1w', 'FLAIR', 'target'], key_add='target', ref='FLAIR'),
        ResampleToMatchd(keys=['target', 't1w', 'FLAIR'], key_dst='target'),
        NormalizeIntensityd(keys=['t1w', 'FLAIR']),
        SelectSliced(keys=['t1w', 'FLAIR', 'target'], dim=2, slice=slice),
        DivisiblePadd(keys=['t1w', 'FLAIR', 'target'], k=32),
        ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
        ToTensord(keys=['tensor', 'target']),
        CastToTypeD(keys=['tensor', 'target'], dtype=[torch.float, torch.long]),
        ToDeviced(keys=['tensor', 'target'], device=device)
    ])

    return loader(item)


def load_data(path: str,
              device: Optional[torch.device] = None,
              slice: Optional[int] = None):
    item = get_image_files(path)
    item = load_item(item, device, slice)

    return item
