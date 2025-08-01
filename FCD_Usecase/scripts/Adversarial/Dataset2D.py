import glob
import os.path

import torch
from monai.transforms import Compose, ToTensord
from torch.utils.data import Dataset
from tqdm import tqdm

from FCD_Usecase.scripts.utils.CustomTransforms import AddMissingd
from FCD_Usecase.scripts.utils.utils import load_image

exceptions = ['sub-00002',
              'sub-00074',
              'sub-00130',
              'sub-00120',
              'sub-00027',
              'sub-00018',
              'sub-00053',
              'sub-00112']


class Dataset2D(Dataset):
    def __init__(self, path: str, slice_dim: int = 2, p: float = 0.0):
        super().__init__()
        self.slice_dim = slice_dim + 1
        self.p = p

        self.data = {}
        self.len = 0

        self.add_change = Compose([
            AddMissingd(keys=['tensor'], key_add='change', ref='tensor'),
            ToTensord(keys=['change'])
        ])

        for x in tqdm([x for x in sorted(glob.glob(f"{path}/sub-*"), key=lambda x: int(x[-5:])) if os.path.isdir(x)],
                      desc='loading data'):
            if os.path.basename(x) in exceptions:
                continue

            item, num_slices = self.get_image(x)

            for i in range(num_slices):
                self.data[self.len] = (item, i)
                self.len += 1

    def get_image(self, path: str):
        item = load_image(path)
        item = self.add_change(item)

        num_slices = item['tensor'].shape[self.slice_dim]

        return item, num_slices

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item, i = self.data[index]
        image = item['tensor'].select(self.slice_dim, i)
        target = item['target'].select(self.slice_dim, i)
        change = item['change'].select(self.slice_dim, i)

        if torch.rand(1).item() < self.p:
            change += torch.randn_like(change) * 0.1

        return image, target[0], change

    def __setitem__(self, index, value):
        item, i = self.data[index]
        item['change'].select(self.slice_dim, i).copy_(value)


if __name__ == '__main__':
    dataset = Dataset2D("data/Dataset101_fcd")
    image, target, change = dataset[0]
    print(len(dataset))
    print(image.shape, image.dtype, image.min(), image.max())
    print(target.shape, target.dtype, target.min(), target.max())
    print(change.shape, change.dtype, change.min(), change.max())
