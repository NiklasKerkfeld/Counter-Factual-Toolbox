import glob
import os
from typing import Tuple, Optional

import pandas as pd
import torch
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, ConcatItemsd, \
    ToTensord, CastToTypeD, DeleteItemsd, Spacingd, ResampleToMatchd, CenterSpatialCropd
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from FCD_Usecase.scripts.utils.CustomTransforms import AddMissingd


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


def get_network(configuration: str, fold: int = 0):
    predictor = nnUNetPredictor()

    predictor.initialize_from_trained_model_folder(
        f"models/Dataset101_fcd/nnUNetTrainer__nnUNetPlans__{configuration}",
        str(fold),
        "checkpoint_best.pth")

    net = predictor.network

    return net


def intersection_over_union(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.flatten()
    target = target.flatten()

    if pred.sum() == 0.0 and target.sum() == 0.0:
        return 1.0

    intersection = torch.sum(torch.logical_and(pred, target))

    return (2 * intersection).item() / (torch.sum(pred) + torch.sum(target) + 1e-6).item()

def f1_score(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, int, int, int, int]:
    pred = pred.flatten().int()
    target = target.flatten().int()

    # True positives, false positives, false negatives
    tp = (pred & target).sum().item()
    fp = (pred & (1 - target)).sum().item()
    fn = ((1 - pred) & target).sum().item()
    tn = ((1 - pred) & (1 - target)).sum().item()

    # Avoid division by zero
    if tp + fp == 0 or tp + fn == 0:
        return 0.0, tp, fp, fn, tn

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0, tp, fp, fn, tn

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, tp, fp, fn, tn


def get_max_slice(target: torch.Tensor, dim: int) -> Tuple[int, int]:
    sizes = torch.sum(target, dim=[i for i in range(target.dim()) if i != dim])
    slice_idx, maximum = sizes.argmax().item(), sizes.max().item()
    if maximum == 0:
        slice_idx = target.shape[dim] // 2
    return slice_idx, maximum


def get_split():
    split = {}
    df = pd.read_csv('data/Dataset101_fcd/participants.tsv', sep='\t')
    df = df[df['mri_diagnosis'] == 'suspicion']
    train = list(df[df['split'] == 'train']['participant_id'])
    split['train'] = train[14:]
    split['valid'] = train[:14]
    split['test'] = list(df[df['split'] == 'test']['participant_id'])

    return split


def get_image(path, slice_dim, slice_idx: Optional[int] = None):
    item = load_image(path)
    if slice_idx is None:
        slice_idx, size = get_max_slice(item['target'], slice_dim + 1)
        print(f"selected slice: {slice_idx} with a target size of {size} pixels.")
    image = item['tensor'].select(slice_dim + 1, slice_idx)[None]
    target = item['target'].select(slice_dim + 1, slice_idx)
    return image, target

