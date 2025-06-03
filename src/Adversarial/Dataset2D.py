import glob
import os.path
from typing import Tuple, Dict

import torch
from monai.transforms import Compose, LoadImaged, ResampleToMatchd, NormalizeIntensityd, \
    ConcatItemsd, ToTensord, CenterSpatialCropd, DeleteItemsd, CastToTyped
from torch.utils.data import Dataset
from tqdm import tqdm

from src.Architecture.CustomTransforms import AddMissingd
from src.utils import get_image_files

exceptions = ['sub-00002',
              'sub-00074',
              'sub-00130',
              'sub-00120',
              'sub-00027',
              'sub-00018',
              'sub-00053',
              'sub-00112']


def load_image(path: str, slice_dim: int = 0) -> Tuple[Dict[str, torch.Tensor], int]:
    load = Compose([
        LoadImaged(keys=['t1w', 'FLAIR', 'target'],
                   reader="NibabelReader",
                   ensure_channel_first=True,
                   allow_missing_keys=True),
        AddMissingd(keys=['FLAIR'], key_add='target', ref='FLAIR'),
        ResampleToMatchd(keys=['target', 't1w', 'FLAIR'], key_dst='target')
    ])

    normalize = NormalizeIntensityd(keys=['t1w', 'FLAIR'])

    preprocess = Compose([
        CenterSpatialCropd(keys=['t1w', 'FLAIR', 'target'], roi_size=(160, 256, 256)),
        ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
        AddMissingd(keys=['tensor'], key_add='change', ref='tensor'),
        ToTensord(keys=['tensor', 'target', 'change']),
        CastToTyped(keys=['target'], dtype=torch.long),
        DeleteItemsd(keys=['t1w', 'FLAIR'])
    ])

    try:
        item = load(get_image_files(path))
        item = normalize(item)
        item = preprocess(item)
    except Exception as e:
        print(f"While loading this file: {path} an Error occurred.")
        raise e

    num_slices: int = item['tensor'].shape[slice_dim]

    return item, num_slices


class Dataset2D(Dataset):
    def __init__(self, path: str, slice_dim: int = 0):
        super().__init__()

        self.slice_dim = slice_dim + 1

        self.data = {}
        self.len = 0
        for x in tqdm([x for x in glob.glob(f"{path}/sub-*")[:5] if os.path.isdir(x)],
                      desc='loading data'):
            if os.path.basename(x) in exceptions:
                continue
            item, num_slices = load_image(x, self.slice_dim)
            for i in range(num_slices):
                self.data[self.len] = (item, i)
                self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item, i = self.data[index]
        image = item['tensor'].select(self.slice_dim, i)
        target = item['target'].select(self.slice_dim, i)
        change = item['change'].select(self.slice_dim, i)

        return image, target[0], change

    def __setitem__(self, index, value):
        item, i = self.data[index]
        item['change'].select(self.slice_dim, i).copy_(value)


if __name__ == '__main__':
    dataset = Dataset2D("data/Dataset101_fcd")
    image, target = dataset[0]
    print(len(dataset))
    print(image.shape)
    print(target.shape)
