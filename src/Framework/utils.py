import glob
import os
from typing import Mapping, Hashable, Optional, Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import LoadImaged, Compose, ResampleToMatchd, ToTensord, ConcatItemsd, \
    MapTransform, ToDeviced, DivisiblePadd, NormalizeIntensityd
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


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


def get_network(configuration: str, fold: int = 0):
    predictor = nnUNetPredictor()

    predictor.initialize_from_trained_model_folder(
        f"models/Dataset101_fcd/nnUNetTrainer__nnUNetPlans__{configuration}",
        str(fold),
        "checkpoint_best.pth")

    net = predictor.network

    return net


def get_image_files(path: str):
    name = os.path.basename(path)
    item = {
        't1w': glob.glob(f"{path}/anat/{name}*T1w.nii.gz")[0],
        'FLAIR': glob.glob(f"{path}/anat/{name}*FLAIR.nii.gz")[0]
    }
    roi_paths = list(glob.glob(f"{path}/anat/{name}*FLAIR_roi.nii.gz"))
    if len(roi_paths) > 0:
        item['target'] = roi_paths[0]

    return item


def load_item(item: Dict[str, str],
              device: Optional[torch.device] = None,
              slice: Optional[int] = None):

    device = device if device is not None else 'cpu'

    loader = Compose([
        LoadImaged(keys=['t1w', 'FLAIR', 'target'],
                   reader="NibabelReader",
                   ensure_channel_first=True,
                   allow_missing_keys=True),
        AddMissingd(keys=['t1w', 'FLAIR', 'target'], key_add='target', ref='FLAIR'),
        ResampleToMatchd(keys=['target', 't1w', 'FLAIR'], key_dst='target'),
        NormalizeIntensityd(keys=['t1w', 'FLAIR']),
        SelectSliced(keys=['t1w', 'FLAIR', 'target'], dim=2, slice=slice),
        DivisiblePadd(keys=['t1w', 'FLAIR', 'target'], k=32),
        ConcatItemsd(keys=['t1w', 'FLAIR'], name='tensor', dim=0),
        ToTensord(keys=['tensor', 'target']),
        ToDeviced(keys=['tensor', 'target'], device=device)
    ])

    return loader(item)


def load_data(path: str,
              device: Optional[torch.device] = None,
              slice: Optional[int] = None):
    item = get_image_files(path)
    item = load_item(item, device, slice)

    return item


def plot_results(t1w_image: torch.Tensor, roi: torch.Tensor, pred: torch.Tensor,
                 slice: Optional[int]):
    """
    Saves a figure with three subplots:
    1. Ground truth ROI over T2-weighted image
    2. Prediction over T2-weighted image
    3. Both ROI and prediction over T2-weighted image

    Args:
        t1w_image (torch.Tensor): T1-weighted image tensor with shape [1, H, W, D]
        roi (torch.Tensor): Ground truth ROI tensor with shape [1, H, W, D]
        pred (torch.Tensor): Prediction tensor with shape [1, H, W, D]
        slice (int): Slice index
    """
    if slice is None:
        t1w_image = np.expand_dims(t1w_image, axis=-1)
        roi = np.expand_dims(roi, axis=-1)
        pred = np.expand_dims(pred, axis=-1)
        slice = 0

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Ground truth overlay
    axs[0].imshow(t1w_image[0, :, :, slice], cmap='gray')
    axs[0].imshow(roi[0, :, :, slice], cmap='Greens', alpha=0.3)
    axs[0].set_title('Ground Truth ROI')
    axs[0].axis('off')

    # 2. Prediction overlay
    axs[1].imshow(t1w_image[0, :, :, slice], cmap='gray')
    axs[1].imshow(pred[0, :, :, slice], cmap='Reds', alpha=0.3)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    # 3. Both overlays
    axs[2].imshow(t1w_image[0, :, :, slice], cmap='gray')
    axs[2].imshow(roi[0, :, :, slice], cmap='Greens', alpha=0.3)
    axs[2].imshow(pred[0, :, :, slice], cmap='Reds', alpha=0.3)
    axs[2].set_title('ROI + Prediction')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig("results/pred.png")
    plt.close()


def dice(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = torch.sum(torch.logical_and(pred, target))

    return (2 * intersection) / (torch.sum(pred) + torch.sum(target))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    net = get_network(configuration='3d_fullres', fold=0)
    net = net.to('cuda:0')

    item = load_data("nnUNet/nnUNet_raw/Dataset101_fcd/sub-00002",
                     device=torch.device("cuda:0"),
                     slice=None)

    print(item['tensor'].shape)
    result = net.forward(item['tensor'][None])
    result = result.cpu().detach().numpy()
    result = result.argmax(axis=1)
    print(result.shape)

    plot_results(item['t1w'].numpy(), item['roi'].cpu().numpy(), result, slice=None)
