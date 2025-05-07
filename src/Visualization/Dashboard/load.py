import glob
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from monai.transforms import Compose, LoadImaged, ResampleToMatchd

from src.Framework.utils import AddMissingd


class Loader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_paths = sorted(
            [path for path in glob.glob(f"{dataset_path}/*") if os.path.isdir(path)])

        self.dataset_folders = [os.path.basename(path) for path in self.dataset_paths]

        self.value_data = self.load_dataset(self.dataset_paths)

        self.mri_loader = Compose([
            LoadImaged(keys=['t1w', 'FLAIR', 'target'],
                       reader="NibabelReader",
                       ensure_channel_first=True,
                       allow_missing_keys=True),
            AddMissingd(keys=['t1w', 'FLAIR', 'target'], key_add='target', ref='FLAIR'),
            ResampleToMatchd(keys=['t1w', 'FLAIR'], key_dst='target')])

        self.images = {}
        self.changes = {}
        self.preds = {}
        self.steps = {}

    def reload(self):
        self.dataset_paths = sorted(
            [path for path in glob.glob(f"{self.dataset_path}/*") if os.path.isdir(path)])

        self.value_data = self.load_dataset(self.dataset_paths)

        self.images = {}
        self.changes = {}
        self.preds = {}
        self.steps = {}

    @property
    def runs(self):
        return self.dataset_folders

    def get_image(self, folder: str):
        if folder not in self.images.keys():
            self.images[folder] = self.load_image(f"{self.dataset_path}/{folder}")

        return self.images[folder]

    def get_change(self, folder: str, step: int, sequence: str):
        if folder not in self.changes.keys():
            self.changes[folder] = self.load_changes(f"{self.dataset_path}/{folder}")

        return self.changes[folder][step][sequence]

    def get_pred(self, folder: str, step: int):
        if folder not in self.preds.keys():
            self.preds[folder] = self.load_preds(f"{self.dataset_path}/{folder}")

        return self.preds[folder][step]

    def get_steps(self, folder: str):
        if folder not in self.steps.keys():
            change_files = glob.glob(f"{self.dataset_path}/{folder}/change_*.npz")
            if change_files:
                self.steps[folder] = [int(os.path.basename(file)[7:-4]) for file in change_files]
            else:
                self.steps[folder] = []

        return self.steps[folder]

    @staticmethod
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

    def load_image(self, folder: str) -> Dict[str, np.ndarray]:
        with open(f"{folder}/logs.json") as f:
            item = json.load(f)

        images = self.mri_loader(item['images'])

        return {key: value[0].numpy() for key, value in images.items()}

    @staticmethod
    def load_preds(folder: str) -> Dict[int, np.ndarray]:
        preds: Dict[int, np.ndarray] = {}
        for file in glob.glob(f'{folder}/pred_*.npy'):
            nr = int(os.path.basename(file)[5:-4])
            preds[nr] = np.load(file)[0]

        return preds

    @staticmethod
    def load_changes(folder: str) -> Dict[int, Dict[str, np.ndarray]]:
        changes: Dict[int, Dict[str, np.ndarray]] = {}
        for file in glob.glob(f'{folder}/*.npz'):
            nr = int(os.path.basename(file)[7:-4])
            data = np.load(file)
            changes[nr] = {key: data[key] for key in data.files}

        return changes


loader = Loader("logs/Logger")
