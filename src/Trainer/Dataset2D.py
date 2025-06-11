import os.path
from typing import Literal, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import load_image, get_split

exceptions = ['sub-00002',
              'sub-00074',
              'sub-00130',
              'sub-00120',
              'sub-00027',
              'sub-00018',
              'sub-00053',
              'sub-00112']


class Dataset2D(Dataset):
    def __init__(self,
                 path: str,
                 slice_dim: int = 2,
                 mode: Literal['train', 'valid', 'test'] = 'train'):
        super().__init__()

        self.slice_dim = slice_dim + 1
        self.mode = mode

        self.data = {}
        self.len = 0

        split = get_split()
        for x in tqdm(sorted([f"{path}/{patient}" for patient in split[mode]], key=lambda x: int(x[-5:])),
                      desc='loading data'):
            if os.path.basename(x) in exceptions:
                print('found exception')
                continue

            item, num_slices = self.get_image(x)

            for i in range(num_slices):
                self.data[self.len] = (item, i)
                self.len += 1

    def get_image(self, path: str):
        item = load_image(path)
        num_slices = item['tensor'].shape[self.slice_dim]

        return item, num_slices

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item, i = self.data[index]
        image = item['tensor'].select(self.slice_dim, i)
        target = item['target'].select(self.slice_dim, i)

        return image, target[0]

    def __setitem__(self, index, value):
        item, i = self.data[index]
        item['change'].select(self.slice_dim, i).copy_(value)


if __name__ == '__main__':
    dataset = Dataset2D("data/Dataset101_fcd", mode='valid')
    print(len(dataset))

