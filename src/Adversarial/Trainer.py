import argparse
import copy
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.multiprocessing import get_context

from tqdm import tqdm

from src.Adversarial.Dataset2D import Dataset2D
from src.Architecture.AdversarialGenerator import AdversarialGenerator
from src.Architecture.CustomLoss import MaskedCrossentropy
from src.utils import normalize, get_network

EXAMPLE = 722

def worker_loop(device, job_queue, result_queue, dataset, base_generator, steps):
    generator = copy.deepcopy(base_generator).to(device)

    while True:
        job_idx = job_queue.get()
        if job_idx is None:
            break  # shutdown signal

        image, target, _ = dataset[job_idx]
        image = image[None].to(device)
        target = target[None].to(device)

        generator.reset()
        optimizer = torch.optim.Adam([generator.change], lr=1e-1)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = generator(image, target)
            loss.backward()
            optimizer.step()

        dataset[job_idx][2].copy_(generator.change.data.cpu())
        result_queue.put(loss.detach().cpu().item())



class Trainer:
    def __init__(self, iterations: int, epochs: int, steps: int, name: str):
        self.iterations = iterations
        self.epochs = epochs
        self.steps = steps
        self.name = name

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = get_network(configuration='2d', fold=0)
        self.generator = AdversarialGenerator(self.model, (2, 256, 256), loss=MaskedCrossentropy())
        self.generator.to(self.device)
        self.loss_fn = torch.nn.MSELoss()

        self.dataset = Dataset2D("data/Dataset101_fcd")
        self.dataloader_gen = DataLoader(self.dataset)
        self.dataloader_train = DataLoader(self.dataset,
                                           batch_size=16,
                                           shuffle=True)

        # setup tensorboard
        train_log_dir = f"logs/adversarial/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore
        self.epoch = 0

        image, target, _ = self.dataset[EXAMPLE]
        self.log_image("original",
                       t1w=normalize(image[0]),
                       flair=normalize(image[1]),
                       target=target)

    def train_adversarial(self):
        optimizer = torch.optim.Adam(self.generator.adversarial.parameters(), lr=1e-3)

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
                loss = self.loss_fn(pred, change)
                loss.backward()
                optimizer.step()

                loss_lst.append(loss.detach().cpu().item())
                bar.set_description(f"training loss: {loss.item():.8f}")

            self.save(f"adversarial.pth")

            print(f"finished training epoch {e + 1} with an average loss of {np.mean(loss_lst)}")
            self.log_loss("training", loss=np.mean(loss_lst))

    def generate(self, alpha: float = 1.0):
        self.generator.alpha = alpha
        loss_lst = []

        bar = tqdm(self.dataloader_gen, desc='generating')
        for idx, (image, target, change) in enumerate(bar):
            self.generator.reset()
            optimizer = torch.optim.Adam([self.generator.change], lr=1e-1)

            image = image.to(self.device)
            target = target.to(self.device)

            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = self.generator(image, target)
                loss.backward()
                optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            self.dataset[idx][2].copy_(self.generator.change.data.cpu())

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

    def generate_distributed(self, worker=16, alpha: float = 1.0):
        self.generator.alpha = alpha
        ctx = get_context('spawn')
        job_queue = ctx.Queue()
        result_queue = ctx.Queue()

        numb_gpus = torch.cuda.device_count()
        gpus = [torch.device(f'cuda:{i}') for i in range(numb_gpus)]

        base_generator = copy.deepcopy(self.generator).cpu()

        workers = []
        for i in range(worker):
            p = ctx.Process(
                target=worker_loop, # worker_id, job_queue, result_queue, dataset, base_generator, steps
                args=(gpus[i % numb_gpus], job_queue, result_queue, self.dataset, base_generator, self.steps)
            )
            p.start()
            workers.append(p)

        for idx in range(len(self.dataset)):
            job_queue.put(idx)

        for _ in range(worker):
            job_queue.put(None)

        loss_lst = []
        for _ in tqdm(range(len(self.dataset)), desc="collecting results"):
            loss_lst.append(result_queue.get())

        for p in workers:
            p.join()

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
        for i in range(self.iterations):
            self.generate_distributed(alpha=min([float(i), 1.0]))
            self.train_adversarial()

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs("models/", exist_ok=True)
        torch.save(
            self.generator.adversarial.state_dict(),
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
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of epochs",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=3,
        help="Number of epochs",
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=15,
        help="Number of epochs",
    )

    return parser.parse_args()


if __name__ == '__main__':
    import glob

    args = get_args()

    args.name = f"{len(glob.glob('logs/adversarial/*'))}_{args.name}"
    trainer = Trainer(iterations=args.iterations,
                      epochs=args.epochs,
                      steps=args.steps,
                      name=args.name)

    trainer.train()
