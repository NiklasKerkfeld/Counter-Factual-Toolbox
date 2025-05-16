from typing import Tuple, Optional

import numpy as np
import torch
from monai.data import DataLoader
from monai.networks.nets import BasicUnet
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.Adversarial.AdversarialWrapper import AdversarialWrapper
from src.Adversarial.Dataset import CacheDataset
from src.Framework.utils import get_network


class Trainer:
    def __init__(self,
                 model: AdversarialWrapper,
                 dataset: CacheDataset,
                 logging_path: str = "logs/Adversarial",):
        self.dataset = dataset

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=1e-1)
        self.adv_optimizer = torch.optim.Adam(self.model.adversarial.parameters(), lr=1e-3)

        self.gen_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.MSELoss()

        self.writer = SummaryWriter(log_dir=logging_path)
        self.step = 0

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

            # update step
            self.adv_optimizer.zero_grad()
            _, pred = self.model(image)
            loss = self.adv_loss(pred, target)
            loss.backward()
            self.adv_optimizer.step()

            # logging
            self.log_value("Adversarial", loss=loss.item())
            losses.append(loss.cpu().detach().item())
            self.step += 1

        print(f"finish training with average loss: {np.mean(losses)}")

    def generate_dataset(self):
        # set dataset to generate mode (returns segmentation as target)
        self.dataset.generate_mode()
        self.model.generate_mode()
        self.model.reset()
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        gen_losses, adv_losses, losses = [], [], []
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

            gen_losses.append(gen_loss.detach().cpu().item())
            adv_losses.append(adv_loss.detach().cpu().item())
            losses.append(loss.detach().cpu().item())

            self.dataset.change_change(item['idx'].item(), self.model.change.data)

        self.log_value("Generator", gen_loss=np.array(gen_losses).mean())
        self.log_value("Generator", adv_loss=np.array(adv_losses).mean())
        self.log_value("Generator", loss=np.array(losses).mean())

    def log_value(self, category: str, **kwargs: float) -> None:
        """
        Logs the loss values to tensorboard.

        Args:
            dataset: Name of the dataset the loss comes from ('Training' or 'Valid')
            step: Optional value for step (default is current epoch)
            kwargs: dict with loss names (keys) and loss values (values)

        """
        # logging
        for key, value in kwargs.items():
            self.writer.add_scalar(
                f"{category}/{key}",
                value,
                global_step=self.step
            )  # type: ignore

        self.writer.flush()  # type: ignore


if __name__ == '__main__':
    generator = get_network(configuration='3d_fullres', fold=0)
    adversarial = BasicUnet(spatial_dims=3,
                            features=(32, 32, 64, 128, 256, 32),
                            in_channels=2,
                            out_channels=2)

    model = AdversarialWrapper(generator, adversarial, (2, 160, 256, 256))

    dataset = CacheDataset("data/Dataset101_fcd",
                           "data/change",
                           "data/cache")

    trainer = Trainer(model, dataset)
    trainer.train(10)
