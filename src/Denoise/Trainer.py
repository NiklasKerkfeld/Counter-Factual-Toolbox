import argparse
import os
from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import tqdm

from src.Architecture.LossFunctions import RelativeL1Loss
from src.Denoise.Dataset2D import Dataset2D

EXAMPLE = 417


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 name: str = 'Trainer',
                 epochs: int = 3,
                 batch_size: int = 16,
                 noise: Union[float, Tuple[float, float]] = (0., 1.)):
        self.model = model
        self.epochs = epochs
        self.epoch = 0
        self.name = name
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.loss_fn = RelativeL1Loss()

        self.dataset = Dataset2D("data/Dataset101_fcd", noise=noise)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)

        # setup tensorboard
        train_log_dir = f"logs/denoise/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

    def train(self):
        min_loss = float('inf')

        for e in range(self.epochs):
            print(f"start epoch {e + 1}:")
            self.epoch += 1
            loss_lst = []

            bar = tqdm(self.dataloader)
            for image, change in bar:
                image = image.to(self.device)
                change = change.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(image)
                loss = self.loss_fn(pred, change)
                loss.backward()
                self.optimizer.step()

                loss_lst.append(loss.detach().cpu().item())
                bar.set_description(f"training loss: {loss.item():.8f}")

            mean_loss = np.mean(loss_lst)
            if mean_loss <= min_loss:
                min_loss = mean_loss
                self.save()

            print(f"finished training epoch {e + 1} with an average loss of {mean_loss}")
            self.log_loss("training", loss=mean_loss)

    def save(self) -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs("models/", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"models/{self.name}_denoiser.pth",
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
        default=10,
        help="Number of epochs to train adversarial in every iteration",
    )

    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--noise",
        "-s",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        help="Noise added to the image (e.g., --noise 0.0 1.0)",
    )

    return parser.parse_args()


if __name__ == '__main__':
    from monai.networks.nets import BasicUNet
    import glob

    args = get_args()
    print(args)

    model = BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128))

    args.name = f"{len(glob.glob('logs/adversarial/*'))}_{args.name}"
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        name=args.name,
        batch_size=args.batchsize,
        noise=args.noise)

    trainer.train()
