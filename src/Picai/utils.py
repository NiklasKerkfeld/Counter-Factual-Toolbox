import glob
import os
from typing import List, Dict

from monai.data import CacheDataset
from monai.transforms import Compose, LoadImaged, RandRotated, RandShiftIntensityd, \
    RandScaleIntensityd, ConcatItemsd, ToTensord, ToDeviced, SqueezeDimd, RandGaussianNoised, Transposed


def get_data(path) -> List[Dict[str, str]]:
    folders = glob.glob(f"{path}/**/**")
    data = []
    for folder in folders:
        name = folder.split(os.sep)[-2]
        data.append({
            'adc': f'{folder}/{name}_adc.mha',
            'hbv': f'{folder}/{name}_hbv.mha',
            't2w': f'{folder}/{name}_t2w.mha',
            'lesion': f'{folder}/{name}_lesion.nii.gz'
        })

    return data


def get_dataset(path: str, train_mode: bool, device='cpu'):
    data = get_data(path)

    sequences = ['adc', 'hbv', 't2w']

    transform_list = [
        LoadImaged(keys=sequences + ['lesion'], ensure_channel_first=True),
        ConcatItemsd(keys=sequences, name='tensor', dim=1),
    ]

    if train_mode:
        transform_list.extend([
            RandRotated(keys=['tensor', 'lesion'], range_x=0.0, range_y=0.0,
                        range_z=0.17, prob=0.2,
                        keep_size=True,
                        mode=['bilinear', 'nearest']),
            RandShiftIntensityd(keys=['tensor'], offsets=0.1, prob=0.2),
            RandScaleIntensityd(keys=['tensor'], factors=0.1, prob=0.2),
            RandGaussianNoised(keys=['tensor'], std=.2, prob=0.2)
        ])

    transform_list.extend([
        SqueezeDimd(keys=['tensor', 'lesion'], dim=0),
        SqueezeDimd(keys=['lesion'], dim=0),
        ToTensord(keys=['tensor', 'lesion']),
        ToDeviced(keys=['tensor', 'lesion'], device=device)
    ])

    return CacheDataset(data, Compose(transform_list))
