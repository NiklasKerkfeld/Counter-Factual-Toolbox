import glob
import json
import os
import threading
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from monai.transforms import Compose, LoadImaged, ResampleToMatchd

from src.Architecture.CustomTransforms import AddMissingd


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
        self.scales = {}
        self.changes = {}
        self.preds = {}
        self.steps = {}

        # Lock per folder
        self.image_locks: Dict[str, threading.Lock] = {}
        self.change_locks: Dict[str, threading.Lock] = {}
        self.pred_locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()

    def reload(self):
        self.dataset_paths = sorted(
            [path for path in glob.glob(f"{self.dataset_path}/*") if os.path.isdir(path)])

        self.value_data = self.load_dataset(self.dataset_paths)

        self.images = {}
        self.scales = {}
        self.changes = {}
        self.preds = {}
        self.steps = {}

    @property
    def runs(self):
        return self.dataset_folders

    def get_image(self, folder: str) -> Dict[str, np.ndarray]:
        with self.global_lock:
            self.image_locks[folder] = threading.Lock()

        with self.image_locks[folder]:
            if folder not in self.images.keys():
                print(f"Loading image for folder: {folder}")
                self.images[folder], self.scales[folder] = self.load_image(
                    f"{self.dataset_path}/{folder}")
            else:
                print(f"Using cached image for folder: {folder}")

        return self.images[folder]

    def get_change(self, folder: str, step: int, sequence: str) -> np.ndarray:
        with self.global_lock:
            self.change_locks[folder] = threading.Lock()

        with self.change_locks[folder]:
            if folder not in self.changes.keys():
                print(f"Loading change for folder: {folder}")
                self.changes[folder] = self.load_changes(f"{self.dataset_path}/{folder}")
            else:
                print(f"Using cached change for folder: {folder}")

        return self.changes[folder][step][sequence]

    def get_changed_image(self, folder: str, step: int, sequence: str) -> np.ndarray:
        image = self.get_image(folder)
        change = self.get_change(folder, step, sequence)

        return image[sequence] + (change * self.scales[folder][sequence][1])

    def get_pred(self, folder: str, step: int) -> np.ndarray:
        with self.global_lock:
            self.pred_locks[folder] = threading.Lock()

        with self.pred_locks[folder]:
            if folder not in self.preds.keys():
                print(f"Loading pred for folder: {folder}")
                self.preds[folder] = self.load_preds(f"{self.dataset_path}/{folder}")
            else:
                print(f"Using cached pred for folder: {folder}")

        return self.preds[folder][step]

    def get_steps(self, folder: str) -> List[int]:
        if folder not in self.steps.keys():
            change_files = glob.glob(f"{self.dataset_path}/{folder}/change_*.npz")
            if change_files:
                self.steps[folder] = [int(os.path.basename(file)[7:-4]) for file in change_files]
            else:
                self.steps[folder] = []

        return self.steps[folder]

    @staticmethod
    def load_dataset(folders) -> pd.DataFrame:
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

    def load_image(self, folder: str) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
        with open(f"{folder}/logs.json") as f:
            item = json.load(f)

        images = self.mri_loader(item['images'])
        image = {key: value[0].numpy() for key, value in images.items()}
        scaling = {key: (value.mean(), value.std()) for key, value in image.items()}

        print(f"loaded images from {folder}")
        return image, scaling

    @staticmethod
    def load_preds(folder: str) -> Dict[int, np.ndarray]:
        preds: Dict[int, np.ndarray] = {}
        for file in glob.glob(f'{folder}/pred_*.npy'):
            nr = int(os.path.basename(file)[5:-4])
            preds[nr] = np.load(file)[0]

        print(f"loaded predictions from {folder}.")
        return preds

    def load_changes(self, folder: str) -> Dict[int, Dict[str, np.ndarray]]:
        changes: Dict[int, Dict[str, np.ndarray]] = {}
        for file in glob.glob(f'{folder}/*.npz'):
            nr = int(os.path.basename(file)[7:-4])
            data = np.load(file)
            changes[nr] = {key: data[key] for key in data.files}

        print(f"loaded changes from {folder}.")
        return changes


loader = Loader("logs/Logger")
