"""Trainer class to train an Unet segmentation model."""

from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import tqdm


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 trainloader: torch.utils.data.DataLoader,
                 testloader: torch.utils.data.DataLoader,
                 optimizer: Optimizer,
                 loss_fn: nn.Module,
                 device: torch.device,
                 name: str = 'Model'):

        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.name = name

        self.step = 0

        # setup tensorboard
        train_log_dir = f"logs/Trainer/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

    def train(self, epochs: int = 10):
        self.model.to(self.device)

        best_loss = float('inf')
        for e in range(1, epochs + 1):
            print(f"start epoch {e}")
            train_loss = self.train_epoch()
            self.log_value('train/loss', train_loss, e)

            valid_loss = self.valid()
            self.log_value('valid/loss', valid_loss, e)

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.model.save(f"{self.name}_es")

        self.model.save(f"{self.name}_end")

    def train_epoch(self) -> float:
        self.model.train()

        losses = []

        for batch in (bar := tqdm(self.trainloader, total=len(self.trainloader))):
            self.step += 1

            images = batch['tensor']
            masks = batch['lesion'].long()

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, masks)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            self.log_value('train/step_loss', loss.item())
            bar.set_description(f"loss: {loss.item()}")

            del images, masks, loss

        return torch.mean(torch.tensor(losses)).item()

    def valid(self) -> float:
        self.model.eval()

        losses = []
        for batch in self.testloader:
            images = batch['tensor']
            masks = batch['lesion'].long()

            with torch.no_grad():
                output = self.model(images)
                loss = self.loss_fn(output, masks)
                losses.append(loss.item())

            del images, masks, loss

        return torch.mean(torch.tensor(losses)).item()

    def log_value(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Logs the loss values to tensorboard.

        Args:
            name: name or title for value on tensorboard
            value: loss values (values)

        """
        # logging
        self.writer.add_scalar(
            f"{name}",
            value,
            global_step=step if step is not None else self.step
        )  # type: ignore

        self.writer.flush()  # type: ignore
