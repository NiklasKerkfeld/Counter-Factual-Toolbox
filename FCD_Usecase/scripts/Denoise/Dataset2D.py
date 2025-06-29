import glob
import os.path
from typing import Tuple, Union

import torch
from monai.transforms import Compose, ToTensord, DeleteItemsd, LoadImaged, Spacingd, \
    ResampleToMatchd, NormalizeIntensityd, ConcatItemsd, CenterSpatialCropd, CastToTypeD
from torch.utils.data import Dataset
from tqdm import tqdm

exceptions = ['sub-00002',
              'sub-00074',
              'sub-00130',
              'sub-00120',
              'sub-00027',
              'sub-00018',
              'sub-00053',
              'sub-00112']

load = Compose([
    LoadImaged(keys=['t1w', 'FLAIR'],
               reader="NibabelReader",
               ensure_channel_first=True,
               allow_missing_keys=True),
    Spacingd(keys=['FLAIR'], pixdim=(1., 1., 1.)),
    ResampleToMatchd(keys=['t1w', 'FLAIR'], key_dst='FLAIR'),
    NormalizeIntensityd(keys=['t1w', 'FLAIR']),
    ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
    ToTensord(keys=['tensor']),
    CenterSpatialCropd(keys=['tensor'], roi_size=(160, 256, 256)),
    CastToTypeD(keys=['tensor'], dtype=[torch.float]),
    DeleteItemsd(keys=['t1w', 'FLAIR']),
    ToTensord(keys=['tensor'])
])


class Dataset2D(Dataset):
    def __init__(self, path: str, slice_dim: int = 2, noise: Union[float, Tuple[float, float]] = (0., 1.)):
        super().__init__()
        self.slice_dim = slice_dim + 1
        self.noise = noise
        self.factor = self.noise if isinstance(noise, float) else None

        self.data = {}
        self.len = 0

        self.add_change = Compose([
            DeleteItemsd(keys=['target']),
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
        name = os.path.basename(path)
        item = {
            't1w': glob.glob(f"{path}/anat/{name}*T1w.nii.gz")[0],
            'FLAIR': glob.glob(f"{path}/anat/{name}*FLAIR.nii.gz")[0]
        }

        item = load(item)
        num_slices = item['tensor'].shape[self.slice_dim]

        return item, num_slices

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item, i = self.data[index]
        image = item['tensor'].select(self.slice_dim, i)

        factor = self.factor if self.factor is not None else self.noise[0] + torch.rand(1).item() * (self.noise[1] - self.noise[0])
        change = torch.randn_like(image) * factor

        return image + change, change

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
