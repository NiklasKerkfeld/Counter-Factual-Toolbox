import glob
import os
from typing import Tuple

import numpy as np
import torch
from monai.data import DataLoader
from monai.networks.nets import BasicUnet
from torch import nn
from torch.utils.data import Dataset

from src.Framework.Framework import ModelWrapper
from src.Framework.utils import get_image_files, save, get_network


class ModelWrapper(nn.Module):
    def __init__(self, segmentation: nn.Module, adversarial: nn.Module, input_shape: Tuple[int, int, int]):
        super(ModelWrapper, self).__init__()
        self.generator = segmentation
        self.adversarial = adversarial

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

        self.adversarial.eval()
        for param in self.adversarial.parameters():
            param.requires_grad = False

        self.change = nn.Parameter(torch.zeros(input_shape))

    def get_input(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.change

    def forward(self, x):
        new_image = self.get_input(x)
        segmentation = self.generator(new_image)
        adversarial = self.adversarial(new_image)

        return segmentation, adversarial



class CacheDataset(Dataset):
    def __init__(self, source_folder: str, cache_folder: str):
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

        folders = glob.glob(f"{source_folder}/sub-*")
        self.dataset = []
        for idx, folder in enumerate(folders):
            item = get_image_files(folder)
            item['idx'] = idx
            item['change'] = None
            self.dataset.append(item)

    def change_change(self, idx: int, change: torch.Tensor):
        item = self.dataset[idx]
        save(change, f"{self.cache_folder}/change_{idx}", item['target'], dtype=np.float32)
        item['change'] = f"{self.cache_folder}/change_{idx}.nii.gz"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class Trainer:
    def __init__(self, adversarial: nn.Module,
                 gernerator: ModelWrapper,
                 dataset: CacheDataset):
        self.adversarial = adversarial
        self.generator = gernerator
        self.dataset = dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-1)
        self.adv_optimizer = torch.optim.Adam(self.adversarial.parameters(), lr=1e-3)

        self.gen_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.MSELoss()


    def train(self, epochs: int):
        for epoch in range(epochs):
            self.train_epoch()
            if epoch != epochs - 1:
                self.generate_dataset()

    def train_epoch(self):
        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)

        for batch in dataloader:
            pass
            # TODO: training stuff

    def generate_dataset(self):
        model = ModelWrapper(self.generator, self.adversarial, (160, 256, 256))
        dataloader = DataLoader(self.dataset, batch_size=16, shuffle=False)

        for item in dataloader:
            image = item['tensor'].to(self.device)
            target = item['target'].to(self.device)

            for _ in range(100):
                # process
                self.gen_optimizer.zero_grad()
                segmentation, adversarial = model(image)
                gen_loss = self.gen_loss(segmentation, target)
                adv_loss = self.adv_loss(segmentation, adversarial)
                loss = gen_loss + adv_loss
                loss.backward()
                self.gen_optimizer.step()

            self.dataset.change_change(item['idx'], model.change.data)



if __name__ == '__main__':
    generator = get_network(configuration='3d_fullres', fold=0)
    adversarial = BasicUnet(spatial_dims=3,
                            features=(32, 32, 64, 128, 256, 32),
                            in_channels=2,
                            out_channels=1)

    dataset = CacheDataset("data/Dataset101_fcd",
                           "data/cache")

    item = dataset[0]

    pass