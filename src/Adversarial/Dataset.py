import glob
import os

import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, ResampleToMatchd, NormalizeIntensityd, \
    ConcatItemsd, ToTensord, DeleteItemsd, CenterSpatialCropd
from torch.utils.data import Dataset

from src.Framework.utils import get_image_files, save, AddMissingd

train_transformations = Compose([
    LoadImaged(keys=['t1w', 'FLAIR', 'change'],
               reader="NibabelReader",
               ensure_channel_first=True,
               allow_missing_keys=True),
    ResampleToMatchd(keys=['t1w', 'FLAIR', 'change'], key_dst='change'),
    CenterSpatialCropd(keys=['t1w', 'FLAIR', 'change'], roi_size=(160, 256, 256)),
    NormalizeIntensityd(keys=['t1w', 'FLAIR']),
    ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
    ToTensord(keys=['tensor', 'change']),
    DeleteItemsd(keys=['t1w', 'FLAIR', 'target'])
])

generate_transformations = Compose([
    LoadImaged(keys=['t1w', 'FLAIR', 'target'],
               reader="NibabelReader",
               ensure_channel_first=True,
               allow_missing_keys=True),
    AddMissingd(keys=['t1w', 'FLAIR', 'target'], key_add='target', ref='FLAIR'),
    ResampleToMatchd(keys=['target', 't1w', 'FLAIR'], key_dst='target'),
    CenterSpatialCropd(keys=['t1w', 'FLAIR', 'change'], roi_size=(160, 256, 256)),
    NormalizeIntensityd(keys=['t1w', 'FLAIR']),
    ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
    ToTensord(keys=['tensor', 'target']),
    DeleteItemsd(keys=['t1w', 'FLAIR', 'change'])
])


class CacheDataset(Dataset):
    def __init__(self, source_folder: str,
                 init_folder: str,
                 cache_folder: str):
        self.output_target = True

        self.init_folder = init_folder
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

        folders = glob.glob(f"{source_folder}/sub-*")
        self.dataset = []
        for idx, path in enumerate(folders):
            item = get_image_files(path)
            item['idx'] = idx
            item['change'] = glob.glob(f"{init_folder}/{os.path.basename(path)}/*change.nii.gz")[0]
            self.dataset.append(item)

    def change_change(self, idx: int, change: torch.Tensor):
        item = self.dataset[idx]
        save(change, f"{self.cache_folder}/change_{idx}", item['target'], dtype=np.float32)
        item['change'] = f"{self.cache_folder}/change_{idx}.nii.gz"

    def __len__(self):
        return len(self.dataset)

    def generate(self):
        self.output_target = True

    def train(self):
        self.output_target = False

    def __getitem__(self, idx: int):
        if self.output_target:
            return generate_transformations(self.dataset[idx])

        return train_transformations(self.dataset[idx])


if __name__ == '__main__':
    from pprint import pprint

    dataset = CacheDataset("data/Dataset101_fcd",
                           "data/change",
                           "data/cache")

    item = dataset[0]

    pprint(item)
    print()
