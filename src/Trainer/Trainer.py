import argparse
import os
from typing import Dict

import numpy as np
import torch
from monai.networks.nets import BasicUNet
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import tqdm

from src.Trainer.Dataset2D import Dataset2D
from src.utils import normalize, dice

EXAMPLE = 444


class Trainer:
    def __init__(self, model: nn.Module, name: str, epochs: int, batch_size: int):
        self.epochs = epochs
        self.name = name
        self.batch_size = batch_size
        self.best_loss = float('inf')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = CrossEntropyLoss(weight=torch.tensor([.1, 10.]).to(self.device))

        self.train_dataset = Dataset2D("data/Dataset101_fcd", mode='train')
        self.valid_dataset = Dataset2D("data/Dataset101_fcd", mode='valid')

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset,
                                           batch_size=1,
                                           shuffle=False)

        # setup tensorboard
        train_log_dir = f"logs/trainer/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore
        self.epoch = 0

        image, target = self.valid_dataset[EXAMPLE]
        pred = F.softmax(self.model(image[None].to(self.device)), dim=1)
        self.log_image("original",
                       t1w=normalize(image[0]),
                       flair=normalize(image[1]),
                       target=target,
                       prediction=pred[0, 1])

    def train(self):
        for e in range(self.epochs):
            print(f"start epoch {e + 1}:")
            self.epoch += 1
            train_loss = self.train_epoch()
            valid_loss = self.valid()

            if valid_loss < self.best_loss:
                self.save(self.name)

            print(
                f"finished training epoch {e + 1}!\n"
                f"\ttrain loss: {train_loss}\n"
                f"\tvalidation loss: {valid_loss}\n")

    def train_epoch(self) -> float:
        self.model.train()

        loss_lst = []
        bar = tqdm(self.train_dataloader)
        for image, target in bar:
            image = image.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            bar.set_description(f"training loss: {loss.item():.8f}")
        self.log_loss("training", loss=np.mean(loss_lst))

        return np.mean(loss_lst)

    def valid(self):
        self.model.eval()

        loss_lst = []
        dice_lst = []
        bar = tqdm(self.valid_dataloader)
        for image, target in bar:
            with torch.no_grad():
                image = image.to(self.device)
                target = target.to(self.device)

                pred = self.model(image)
                loss = self.loss_fn(pred, target)

            pred = F.softmax(pred, dim=1)
            dice_lst.append(dice(pred[:, 1], target).cpu().item())
            loss_lst.append(loss.detach().cpu().item())
            bar.set_description(f"validation loss: {loss.item():.8f}")

        self.log_loss("validation", loss=np.mean(loss_lst))
        self.log_loss("validation", dice=np.mean(dice_lst))

        image, target = self.valid_dataset[EXAMPLE]
        pred = F.softmax(self.model(image[None].to(self.device)), dim=1)
        self.log_image("validation", prediction=pred[0, 1])

        return np.mean(loss_lst)

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs("models/", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"models/{name}",
        )

    def log_loss(self, task: str, **kwargs: Dict[str, float]) -> None:
        """
        Logs the loss values to tensorboard.

        Args:
            task: Name of the dataset the loss comes from ('Training' or 'Valid')
            kwargs: dict with loss names (keys) and loss values (values)

        """
        # logging
        for key, value in kwargs.items():
            self.writer.add_scalar(
                f"{task}/{key}",
                value,
                global_step=self.epoch
            )  # type: ignore

        self.writer.flush()  # type: ignore

    def log_image(self, task: str, **kwargs: torch.Tensor) -> None:
        """
        Logs given images under the given dataset label.

        Args:
            task: dataset to log the images under ('Training' or 'Validation')
            kwargs: Dict with names (keys) and images (images) to log
        """
        for key, image in kwargs.items():
            if image.dim() == 2:
                image = image[None]
            print(
                f"logging {task}/{key}: shape:{image.shape}, min:{image.min()}, max:{image.max()}")
            # log in tensorboard
            self.writer.add_image(
                f"{task}/{key}",
                image,  # type: ignore
                global_step=self.epoch,
                dataformats="CHW"
            )  # type: ignore

        self.writer.flush()  # type: ignore


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with call arguments
    """
    parser = argparse.ArgumentParser(description="training")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="test",
        help="Name of the model, for saving and logging",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=3,
        help="Number of epochs",
    )

    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=16,
        help="Batch size",
    )

    return parser.parse_args()


if __name__ == '__main__':
    import glob

    args = get_args()

    args.name = f"{len(glob.glob('logs/trainer/*'))}_{args.name}"

    model = BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128))

    trainer = Trainer(model=model,
                      epochs=args.epochs,
                      name=args.name,
                      batch_size=args.batchsize)

    trainer.train()
