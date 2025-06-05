import glob
import os
from copy import deepcopy
from typing import Tuple

import monai
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import SaveImage, Compose, LoadImaged, NormalizeIntensityd, ConcatItemsd, \
    ToTensord, CastToTypeD, DeleteItemsd, Spacingd, ResampleToMatchd, CenterSpatialCropd
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from src.Architecture.CustomTransforms import AddMissingd


def load_image(path: str):
    load = Compose([
        LoadImaged(keys=['t1w', 'FLAIR', 'target'],
                   reader="NibabelReader",
                   ensure_channel_first=True,
                   allow_missing_keys=True),
        AddMissingd(keys=['FLAIR'], key_add='target', ref='FLAIR'),
        Spacingd(keys=['FLAIR'], pixdim=(1., 1., 1.)),
        ResampleToMatchd(keys=['t1w', 'FLAIR', 'target'], key_dst='FLAIR'),
        NormalizeIntensityd(keys=['t1w', 'FLAIR']),
        ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
        ToTensord(keys=['tensor', 'target']),
        CenterSpatialCropd(keys=['tensor', 'target'], roi_size=(160, 256, 256)),
        CastToTypeD(keys=['tensor', 'target'], dtype=[torch.float, torch.long]),
        DeleteItemsd(keys=['t1w', 'FLAIR'])
    ])

    name = os.path.basename(path)
    item = {
        't1w': glob.glob(f"{path}/anat/{name}*T1w.nii.gz")[0],
        'FLAIR': glob.glob(f"{path}/anat/{name}*FLAIR.nii.gz")[0]
    }
    roi_paths = list(glob.glob(f"{path}/anat/{name}*FLAIR_roi.nii.gz"))
    if len(roi_paths) > 0:
        item['target'] = roi_paths[0]

    return load(item)


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
    plt.imshow(image, cmap='gray')


    plt.quiver(X, Y, dx, dy, color=color, angles='xy', scale_units='xy', scale=1 / scale)

    plt.title(f'Elastic Deformation Field (scale x{scale})')
    plt.axis('off')
    plt.savefig('logs/deformation.png', dpi=1000)
    plt.close()

    print(f"Visualization of the deformation saved to logs/deformation.png")


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


def save(image: torch.Tensor, path: str, example: monai.data.MetaTensor, post_fix: str = 'pred',
         dtype=np.int8):
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

def get_max_slice(target: torch.Tensor, dim: int) -> Tuple[int, int]:
    sizes = torch.sum(target, dim=[i for i in range(target.dim()) if i != dim])
    return sizes.argmax().item(), sizes.max().item()


if __name__ == '__main__':
    item = load_image('data/Dataset101_fcd/sub-00001')
    pass
