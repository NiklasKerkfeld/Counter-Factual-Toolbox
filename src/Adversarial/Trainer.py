import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import tqdm

from src.Adversarial.Dataset2D import Dataset2D
from src.Architecture.Generator import AdversarialGenerator
from src.Architecture.LossFunctions import MaskedCrossEntropyLoss
from src.utils import normalize, get_network

EXAMPLE = 417


class Trainer:
    def __init__(self, iterations: int = 10,
                 name: str = 'Trainer',
                 epochs: int = 3,
                 steps: int = 25,
                 batch_size: int = 16,
                 alpha: float = 1.0,
                 p: float = 0.0):
        self.iterations = iterations
        self.epochs = epochs
        self.steps = steps
        self.name = name
        self.batch_size = batch_size
        self.alpha = alpha

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = get_network(configuration='2d', fold=0)
        self.generator = AdversarialGenerator(self.model, (self.batch_size, 2, 160, 256),
                                              loss=MaskedCrossEntropyLoss())
        self.generator.to(self.device)
        self.loss_fn = torch.nn.L1Loss()

        self.dataset = Dataset2D("data/Dataset101_fcd", p=p)
        self.dataloader_gen = DataLoader(self.dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False)
        self.dataloader_train = DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)

        # setup tensorboard
        train_log_dir = f"logs/adversarial/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore
        self.epoch = 0

        image, target, _ = self.dataset[EXAMPLE]
        pred = F.softmax(self.generator.model(image[None].to(self.device)), dim=1)
        self.log_image("original",
                       t1w=normalize(image[0]),
                       flair=normalize(image[1]),
                       target=target,
                       prediction=pred[0, 1])

    def train_adversarial(self):
        self.generator.load_adversarial(f"models/{self.name}_adversarial.pth")
        optimizer = torch.optim.Adam(self.generator.adversarial.parameters(), lr=1e-3)
        min_loss = float('inf')

        for e in range(self.epochs):
            print(f"start epoch {e + 1}:")
            self.epoch += 1
            loss_lst = []

            bar = tqdm(self.dataloader_train)
            for image, _, change in bar:
                image = image.to(self.device)
                change = change.to(self.device)

                optimizer.zero_grad()
                pred = self.generator.adversarial(image + change)
                loss = self.loss_fn(pred, torch.abs(change))
                loss.backward()
                optimizer.step()

                loss_lst.append(loss.detach().cpu().item())
                bar.set_description(f"training loss: {loss.item():.8f}")

            mean_loss = np.mean(loss_lst)
            if mean_loss <= min_loss:
                min_loss = mean_loss
                self.save()

            print(f"finished training epoch {e + 1} with an average loss of {mean_loss}")
            self.log_loss("training", loss=mean_loss)

    def generate(self, alpha: float = 1.0):
        self.generator.alpha = alpha
        loss_lst = []

        bar = tqdm(self.dataloader_gen, desc='generating')
        for idx, (image, target, _) in enumerate(bar):
            self.generator.reset()
            optimizer = torch.optim.Adam([self.generator.change], lr=1e-2)

            image = image.to(self.device)
            target = target.to(self.device)

            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = self.generator(image, target)
                loss.backward()
                optimizer.step()

            # save generated change in dataset
            change = self.generator.change.data.cpu()
            for i in range(self.batch_size):
                self.dataset[idx * self.batch_size + i][2].copy_(change[i])

            loss_lst.append(loss.detach().cpu().item())

        # logging
        print(f"finished generating with an average loss of {np.mean(loss_lst)}")
        image, _, change = self.dataset[EXAMPLE]
        tensor = image + change
        pred = F.softmax(self.generator.model(tensor[None].to(self.device)), dim=1)
        self.log_loss("generating", loss=np.mean(loss_lst))
        self.log_image("generating",
                       change_t1w=torch.abs(change[0]),
                       change_flair=torch.abs(change[1]),
                       t1w=normalize(tensor[0]),
                       flair=normalize(tensor[1]),
                       prediction=pred[0, 1])

    def train(self):
        self.generate(alpha=0.0)
        for i in range(self.iterations):
            self.train_adversarial()
            if i != self.iterations - 1:
                self.generate(alpha=self.alpha)

    def save(self) -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs("models/", exist_ok=True)
        torch.save(
            self.generator.adversarial.state_dict(),
            f"models/{self.name}_adversarial.pth",
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
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of epochs to train adversarial in every iteration",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=50,
        help="Number of steps for generating image changes",
    )

    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=1.0,
        help="Ratio of adversarial loss to general loss",
    )

    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--noise_prob",
        "-p",
        type=float,
        default=0.0,
        help="Probability of noise being added to adversarial change",
    )

    return parser.parse_args()


if __name__ == '__main__':
    import glob

    args = get_args()
    print(args)

    args.name = f"{len(glob.glob('logs/adversarial/*'))}_{args.name}"
    trainer = Trainer(iterations=args.iterations,
                      epochs=args.epochs,
                      steps=args.steps,
                      name=args.name,
                      batch_size=args.batchsize,
                      alpha=args.alpha,
                      p=args.noise_prob)

    trainer.train()
