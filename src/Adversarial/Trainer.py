from typing import Tuple, Optional

import numpy as np
import torch
from monai.data import DataLoader
from monai.networks.nets import BasicUnet
from torch import nn
from tqdm import tqdm

from src.Adversarial.Dataset import CacheDataset
from src.Framework.utils import get_network


class ModelWrapper(nn.Module):
    def __init__(self,
                 segmentation: nn.Module,
                 adversarial: nn.Module,
                 input_shape: Tuple[int, int, int, int]):
        super(ModelWrapper, self).__init__()
        self.generator = segmentation
        self.adversarial = adversarial
        self.input_shape = input_shape
        self.mode = 'adversarial'

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

        self.change = nn.Parameter(torch.zeros(self.input_shape))

    def get_input(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.change

    def generate_mode(self):
        self.mode = 'generate'
        self.adversarial.eval()
        for param in self.adversarial.parameters():
            param.requires_grad = False

    def train_mode(self):
        self.mode = 'adversarial'
        self.adversarial.eval()
        for param in self.adversarial.parameters():
            param.requires_grad = True

    def reset(self):
        self.change = nn.Parameter(torch.zeros(self.input_shape))

    def forward(self, x) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.mode == 'generate':
            new_image = self.get_input(x)
            segmentation = self.generator(new_image)
            adversarial = self.adversarial(new_image)

            return segmentation, adversarial

        else:
            return None, self.adversarial(x)


class Trainer:
    def __init__(self,
                 model: ModelWrapper,
                 dataset: CacheDataset):
        self.dataset = dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=1e-1)
        self.adv_optimizer = torch.optim.Adam(self.model.adversarial.parameters(), lr=1e-3)

        self.gen_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.MSELoss()

    def train(self, epochs: int):
        for epoch in range(epochs):
            print(f"start epoch {epoch + 1}")
            self.train_epoch()
            if epoch != epochs - 1:
                self.generate_dataset()

    def train_epoch(self):
        # set dataset to training mode (returns change as target)
        self.dataset.train_mode()
        self.model.train_mode()
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        losses = []
        for batch in tqdm(dataloader, desc='train adversarial', total=len(dataloader)):
            image = batch['tensor'].to(self.device)
            target = batch['change'].to(self.device)
            image += target

            print(f"{image.shape=}")
            print(f"{target.shape=}")

            self.adv_optimizer.zero_grad()
            pred = self.model(image)
            loss = self.adv_loss(pred, target)
            loss.backward()
            self.adv_optimizer.step()

            losses.append(loss.cpu().detach().item())

        print(f"finish training with average loss: {np.mean(losses)}")

    def generate_dataset(self):
        # set dataset to generate mode (returns segmentation as target)
        self.dataset.generate_mode()
        self.model.generate_mode()
        self.model.reset()
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        for item in tqdm(dataloader, desc='generate dataset', total=len(dataloader)):
            image = item['tensor'].to(self.device)
            target = item['target'][:, 0].to(self.device)

            for _ in range(50):
                # process
                self.gen_optimizer.zero_grad()
                segmentation, adversarial = self.model(image)
                gen_loss = self.gen_loss(segmentation, target)
                adv_loss = torch.sum(adversarial)
                loss = gen_loss + adv_loss
                loss.backward()
                self.gen_optimizer.step()

            self.dataset.change_change(item['idx'], self.model.change.data)


if __name__ == '__main__':
    generator = get_network(configuration='3d_fullres', fold=0)
    adversarial = BasicUnet(spatial_dims=3,
                            features=(32, 32, 64, 128, 256, 32),
                            in_channels=2,
                            out_channels=2)

    dataset = CacheDataset("data/Dataset101_fcd",
                           "data/change",
                           "data/cache")

    item = dataset[0]

    trainer = Trainer(adversarial, generator, dataset)
    trainer.train(10)

