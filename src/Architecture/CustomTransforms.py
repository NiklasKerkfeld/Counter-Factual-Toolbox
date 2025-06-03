from typing import Mapping, Hashable, Optional

import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

class SelectSliced(MapTransform):
    def __init__(self, keys: KeysCollection, dim: int, slice: Optional[int] = None):
        """
        Select a slice from the input data.
        :param dim: dimension to select.
        :param slice: index of the slice to select. Doesn't select if None.
        """
        super().__init__(keys)
        self.slice = slice
        self.dim = dim + 1 if dim >= 0 else dim

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        if self.slice is None:
            return data

        # create d as dict of data
        d = dict(data)
        # crop the data
        for key in self.key_iterator(d):
            d[key] = d[key].select(self.dim, self.slice)

        return d


class AddMissingd(MapTransform):
    def __init__(self, keys: KeysCollection, key_add: str, ref: str):
        """
        Adds an empty annotation (roi) if missing.
        """
        super().__init__(keys)
        self.key_add = key_add
        self.ref = ref

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        if self.key_add in data:
            return data

        ref_tensor = data[self.ref]

        if isinstance(ref_tensor, MetaTensor):
            # Clone the tensor with metadata and zero the data
            new_tensor = ref_tensor.clone()
            new_tensor.data.zero_()
        else:
            new_tensor = torch.zeros_like(ref_tensor)

        data[self.key_add] = new_tensor
        return data

