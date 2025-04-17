"""
Preprocesses the Data and saves them
"""
import json
import os
from pathlib import Path
from typing import Tuple, Mapping, Hashable, Optional

import torch
from monai.config import KeysCollection
from tqdm import tqdm

import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    ResampleToMatchd,
    Spacingd,
    MapLabelValued,
    MapTransform,
    SpatialCrop,
    SpatialPadd
)


SPACING = (0.3, 0.3, 1.0)
CROPS = (256, 256, 64)
np.random.seed(42)

exclusions = [11050, 11231]


def get_segment_size(segment: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
    """
    calcs upper and lower bound of segment in x, y and z dim
    :param segment: tensor with segment
    :return: xmin, xmax, ymin, ymax, zmin, zmax
    """

    x = torch.sum(segment, dim=(0, 2, 3))
    y = torch.sum(segment, dim=(0, 1, 3))
    z = torch.sum(segment, dim=(0, 1, 2))

    xmin, xmax = torch.where(x)[0][[0, -1]]
    ymin, ymax = torch.where(y)[0][[0, -1]]
    zmin, zmax = torch.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def get_roi(segment: torch.Tensor, edge: Tuple[int, int, int] = (0, 0, 0),
            kdiv: int = 32) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    calculates the center and the size of a roi given the segmentations of the prostate
    :param segment: tensor with the segmentation
    :param edge: space to add between segmentation and edge of roi
    :param kdiv: ensures size of roi is dividable by kdiv
    :return: coordinates of center and size of roi
    """
    xmin, xmax, ymin, ymax, zmin, zmax = get_segment_size(segment)

    center = xmin + ((xmax - xmin) // 2), ymin + ((ymax - ymin) // 2), zmin + ((zmax - zmin) // 2)

    size = torch.tensor([xmax - xmin + 2 * edge[0], ymax - ymin + 2 * edge[1], zmax - zmin + 2 * edge[2]])
    size += kdiv - (size % kdiv)

    return center, tuple(size)


class CutRoiBySegmentationd(MapTransform):
    def __init__(self, keys: KeysCollection, segmentation_key: str,
                 edge: Tuple[int, int, int] = (0, 0, 0), kdiv: int = 32,
                 forced_size: Optional[Tuple[int, int, int]] = None):
        """
        crops data with roi around the given segmentation
        :param keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        :param segmentation_key: key of the segmentation the roi should be around
        :param edge: min number of pixels between segmentation and roi (default 0)
        :param kdiv: every dim of roi is dividable by kdiv (default 32)
        """
        super().__init__(keys)
        self.forced_size = forced_size
        self.segmentation_key = segmentation_key
        self.edge = edge
        self.kdiv = kdiv

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        # create d as dict of data
        d = dict(data)

        # calculate roi and create cropper
        roi_center, roi_size = get_roi(data[self.segmentation_key], self.edge, self.kdiv)
        if self.forced_size is not None:
            roi_size = self.forced_size
        cropper = SpatialCrop(roi_center=roi_center, roi_size=roi_size)

        # crop the data
        for key in self.key_iterator(d):
            d[key] = cropper(d[key])

        return d


def get_positive_slices(annotation: torch.Tensor):
    positive = (annotation.sum((0, 1, 2)) != 0).nonzero()
    return positive[:, 0]


class PositiveSlices(MapTransform):
    def __init__(self, keys: KeysCollection, segmentation_key: str):
        """
        Separates all slices with a lesion.
        :param keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        :param segmentation_key: key of the annotation
        """
        super().__init__(keys)
        self.segmentation_key = segmentation_key

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        # create d as dict of data
        data = dict(data)

        # calculate roi and create cropper
        slice_indices = get_positive_slices(data[self.segmentation_key])

        # crop the data
        slices = []
        for i in slice_indices:
            d = {}
            for key in self.key_iterator(data):
                d[key] = data[key][:, :, :, i]
            slices.append(d)

        return slices


# define monai transformations
preprocess = Compose([
    # Load all imges and add a Channel at beginning (C, X, Y, Z)
    LoadImaged(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], ensure_channel_first=True, allow_missing_keys=True),

    # Clip intensity between 1. and 99. percentile and scale it between 0, 1
    ScaleIntensityRangePercentilesd(keys=['adc', 'hbv', 't2w'], lower=1, upper=99, b_min=0, b_max=1,
                                    allow_missing_keys=True),

    # Normalize intensity
    NormalizeIntensityd(keys=['adc', 'hbv', 't2w'], allow_missing_keys=True),

    # ensure label and prostate only have 0 and 1 values
    MapLabelValued(keys=['lesion'], orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                   target_labels=[0, 0, 1, 1, 1, 1, 1, 1], allow_missing_keys=True),

    MapLabelValued(keys=['prostate'], orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                   target_labels=[0, 1, 1, 1, 1, 1, 1, 1], allow_missing_keys=True),

    # resample images to t2w so all images of patient have same size
    ResampleToMatchd(keys=['adc', 'hbv', 'lesion', 'prostate'], key_dst='t2w',
                     mode=['bilinear', 'bilinear', 'nearest', 'nearest'], allow_missing_keys=True),

    # ensure all images of one patient have same spacing
    Spacingd(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], pixdim=SPACING,
             mode=("bilinear", "bilinear", "bilinear", "nearest", "nearest"), allow_missing_keys=True),

    # cut out Region of prostate of the images
    CutRoiBySegmentationd(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'],
                          segmentation_key='prostate',
                          forced_size=CROPS),
    SpatialPadd(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], spatial_size=CROPS),
    PositiveSlices(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], segmentation_key='lesion')
])


def preprocessing(patient: str, name: str, category: str):
    """
    preprocessing mri images with preprocessing transformation adding lesion mask with zeros if patient is negative
    :param patient: folder name of the patient
    :param name: file name of the images
    :param category: category (train, valid, test) for saving in
    :param positive: true if patient is positive casePCa > 1
    :return:
    """
    item = {
        'adc': f'{Path(__file__).parent.absolute()}/full/{patient}/{name}_adc.mha',
        'hbv': f'{Path(__file__).parent.absolute()}/full/{patient}/{name}_hbv.mha',
        't2w': f'{Path(__file__).parent.absolute()}/full/{patient}/{name}_t2w.mha',
        'prostate': f'{Path(__file__).parent.absolute()}/full/{patient}/{name}_prostate.nii.gz',
        'lesion': f'{Path(__file__).parent.absolute()}/full/{patient}/{name}_lesion.nii.gz'
    }

    # preprocess images
    slices = preprocess(item)

    for i, slice in enumerate(slices):
        path = f"{Path(__file__).parent.absolute()}/preprocessed/{category}/{name}/{i+1}"
        os.makedirs(path, exist_ok=True)

        input_data = torch.concat((slice['t2w'], slice['hbv'], slice['adc']), dim=0)

        np.save(f"{path}/input.npy", np.rot90(input_data.numpy(), k=3, axes=(1, 2)))
        np.save(f"{path}/target.npy", np.rot90(slice['lesion'].numpy(), k=3, axes=(1, 2)))
        np.save(f"{path}/prostate.npy", np.rot90(slice['prostate'].numpy(), k=3, axes=(1, 2)))


def main():
    """
    preprocesses the data and saves the results according to their train, valid, test split
    :return:
    """
    with open(f'{Path(__file__).parent.absolute()}/split.json') as f:
        split = json.load(f)

    positive = split['positive']

    for patient, file_name in tqdm(positive['train'], desc='preprocessing positive training'):
        preprocessing(patient, file_name, 'train')

    for patient, file_name in tqdm(positive['test'], desc='preprocessing positive test'):
        preprocessing(patient, file_name, 'test')

    for patient, file_name in tqdm(positive['valid'], desc='preprocessing positive valid'):
        preprocessing(patient, file_name, 'valid')


if __name__ == '__main__':
    main()
