import json
import os
import time
from typing import Dict, Union

import numpy as np
import torch


class Logger:
    def __init__(self, logging_path: str, images_paths: Dict[str, str], target_path: str):
        self.logging_path = logging_path
        self.modalities = list(images_paths.keys())

        os.makedirs(self.logging_path)
        with open(f"{self.logging_path}/logs.json", 'w') as f:
            json.dump({'name': os.path.basename(logging_path),
                       'time': time.time(),
                       'images': {**images_paths, 'target': target_path}}, f)

        with open(f"{self.logging_path}/loss.csv", "x") as f:
            f.write(f"step,key,value\n")

    def log_change(self, step: int, change: Union[torch.tensor, np.ndarray]):
        change = {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in
                  zip(self.modalities, change)}
        np.savez(f"{self.logging_path}/change_{step}.npz", **change)

    def log_prediction(self, step: int, prediction: Union[torch.tensor, np.ndarray]):
        prediction = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
        np.save(f"{self.logging_path}/pred_{step}.npy", prediction)

    def log_values(self, step: int, **kwargs):
        with open(f"{self.logging_path}/loss.csv", 'a') as file:
            for key, value in kwargs.items():
                file.write(f"{step},{key},{value}\n")


if __name__ == '__main__':
    images = {
        't1w': "nnUNet/nnUNet_raw/Dataset101_fcd/sub-00001/anat/sub-00001_acq-iso08_T1w.nii.gz",
        'FLAIR': 'nnUNet/nnUNet_raw/Dataset101_fcd/sub-00001/anat/sub-00001_acq-T2sel_FLAIR.nii.gz'
    }
    target = 'nnUNet/nnUNet_raw/Dataset101_fcd/sub-00001/anat/sub-00001_acq-T2sel_FLAIR_roi.nii.gz'

    logger = Logger('logs/new/run3', images, target)

    for i in range(1, 6):
        change = {
            't1w': torch.randn((160, 256, 256)) * 10,
            'FLAIR': torch.randn((160, 256, 256)),
        }
        logger.log_change(change, step=i)

        pred = torch.randint(0, 10_000, (160, 256, 256)) / 10_000
        logger.log_prediction(pred, step=i)

        logger.log_values(step=i, loss=np.random.randn(1).item() ** 2,
                          acc=np.random.randn(1).item() ** 2)

    for i in range(10):
        logger.log_values(step=i, test=np.random.randn(1).item() ** 2)
