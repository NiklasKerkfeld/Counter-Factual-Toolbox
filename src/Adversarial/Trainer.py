from typing import Tuple

import numpy as np
import torch
from monai.data import DataLoader
from monai.networks.nets import BasicUnet
from torch import nn
from tqdm import tqdm

from src.Adversarial.Dataset import CacheDataset
from src.Framework.utils import get_network, get_vram


class ModelWrapper(nn.Module):
    def __init__(self, segmentation: nn.Module, adversarial: nn.Module, input_shape: Tuple[int, int, int, int]):
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


class Trainer:
    def __init__(self,
                 adversarial: nn.Module,
                 gernerator: ModelWrapper,
                 dataset: CacheDataset):
        self.dataset = dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.adversarial = adversarial.to(self.device)
        self.generator = gernerator.to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-1)
        self.adv_optimizer = torch.optim.Adam(self.adversarial.parameters(), lr=1e-3)

        self.gen_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.MSELoss()

        used, total, free = get_vram(self.device)
        print(f"loaded models currently {used} GB out of {total} GB of VRAM in use. {free} GB left!")

    def train(self, epochs: int):
        for epoch in range(epochs):
            print(f"start epoch {epoch + 1}")
            self.train_epoch()
            if epoch != epochs - 1:
                self.generate_dataset()

    def train_epoch(self):
        # set dataset to training mode (returns change as target)
        self.dataset.train()
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        losses = []
        for batch in tqdm(dataloader, desc='train adversarial', total=len(dataloader)):
            image = batch['tensor'].to(self.device)
            target = batch['change'].to(self.device)
            image += target

            self.adv_optimizer.zero_grad()
            pred = self.adversarial(image)
            loss = self.adv_loss(pred, target)
            loss.backward()
            self.adv_optimizer.step()

            losses.append(loss.cpu().detach().item())

        print(f"finish training with average loss: {np.mean(losses)}")

    def generate_dataset(self):
        # set dataset to generate mode (returns segmentation as target)
        self.dataset.generate()
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        model = ModelWrapper(self.generator, self.adversarial, (2, 160, 256, 256))
        model.to(self.device)

        for item in tqdm(dataloader, desc='generate dataset', total=len(dataloader)):
            image = item['tensor'].to(self.device)
            target = item['target'].to(self.device)

            for _ in range(100):
                # process
                self.gen_optimizer.zero_grad()
                segmentation, adversarial = model(image)
                print(f"{segmentation.shape=}")
                print(f"{target.shape=}")
                print(f"{adversarial.shape=}")
                gen_loss = self.gen_loss(segmentation, target)
                adv_loss = torch.sum(adversarial)
                loss = gen_loss + adv_loss
                loss.backward()
                self.gen_optimizer.step()

            self.dataset.change_change(item['idx'], model.change.data)


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

