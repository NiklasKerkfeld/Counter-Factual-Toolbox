import glob
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from monai.transforms import Compose, LoadImaged, ResampleToMatchd

from src.Framework.utils import normalize
from src.fcd.utils import AddMissingd


def load_dataset(folders):
    # Load dataset folders dynamically
    all_data = []

    for folder in folders:
        path = f"{folder}/loss.csv"

        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['dataset'] = os.path.basename(folder)  # Add a column to track source
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df


dataset_path = "logs/new"  # <-- Change this to your dataset folder
dataset_paths = sorted([path for path in glob.glob(f"{dataset_path}/*") if os.path.isdir(path)])
dataset_folders = [os.path.basename(path) for path in dataset_paths]

value_data = load_dataset(dataset_paths)


mri_loader = Compose([
    LoadImaged(keys=['t1w', 'FLAIR', 'target'],
               reader="NibabelReader",
               ensure_channel_first=True,
               allow_missing_keys=True),
    AddMissingd(keys=['t1w', 'FLAIR', 'target'], key_add='target', ref='FLAIR'),
    ResampleToMatchd(keys=['t1w', 'FLAIR'], key_dst='target')])


def load_image(folder: str) -> Dict[str, np.ndarray]:
    with open(f"{folder}/logs.json") as f:
        item = json.load(f)

    images = mri_loader(item['images'])

    return {key: normalize(np.rot90(value[0].numpy(), k=3, axes=(0, 2))) for key, value in images.items()}


images = {folder: load_image(path) for folder, path in zip(dataset_folders, dataset_paths)}
print(images['run1']['t1w'].shape)