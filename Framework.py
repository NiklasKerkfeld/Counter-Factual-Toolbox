from typing import Tuple, List

import torch
from torch.func import grad_and_value
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import trange

from dataset import DummyDataset
from model import SimpleUNet
from utils import symetric_color_mapping


class Loss(nn.Module):
    def __init__(self, loss_fn, beta: float = 1., gamma: float = 1.):
        super().__init__()
        self.loss_fn = loss_fn
        self.beta = beta
        self.gamma = gamma
        self.relu = nn.ReLU()

        self.grad_x = torch.tensor([[[[1., 0., -1.],
                                       [2., 0., -2.],
                                       [1., 0., -1.]]]], requires_grad=False)

        self.grad_y = torch.tensor([[[[1., 2., 1.],
                                       [0., 0., 0.],
                                       [-1., -2., -1.]]]], requires_grad=False)

    def to(self, device: torch.device):
        self.grad_x = self.grad_x.to(device)
        self.grad_y = self.grad_y.to(device)

    def magnitude(self, change):
        dx = torch.nn.functional.conv2d(change[None], self.grad_x)
        dy = torch.nn.functional.conv2d(change[None], self.grad_y)
        return torch.mean((torch.abs(dx) + torch.abs(dy)))

    def forward(self, pred, input, target, change):
        # normal loss
        loss = self.loss_fn(pred, target)

        # regularize to achieve small changes
        reg = torch.mean(torch.abs(change))

        # penalize values out of image range
        over = torch.sum(self.relu(input - 1))
        under = torch.sum(self.relu(-input))

        # penalize different neighbors
        magnitude = self.magnitude(change)

        return loss + self.beta * reg + over + under + self.gamma * magnitude


class Framework:
    def __init__(self, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 input_shape: Tuple[int, int, int],
                 device: torch.device,
                 name: str = "framework",
                 lr: float = 0.1,
                 decay: float = 1.0,
                 ):

        self.model = model
        self.model.requires_grad = False

        self.optimizer = optimizer
        self.loss_fn = Loss(loss_fn)

        self.change = torch.zeros(input_shape)
        self.lr = lr
        self.decay = decay

        self.device = device
        self.name = name
        self.step = 0

        # set up grad function
        def func(change, image, mask):
            x = image + change
            pred = self.model(x)
            loss = self.loss_fn(pred, x, mask, change)
            return loss

        self.grad_func = grad_and_value(func)

        # setup tensorboard
        train_log_dir = f"logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

    def process(self, image: torch.Tensor, mask: torch.Tensor) -> None:
        self.model.to(self.device)
        image = image.to(self.device)
        mask = mask.to(self.device)
        self.change = self.change.to(self.device)
        self.loss_fn.to(self.device)

        self.log_image("image", image[0])
        self.log_image("mask", mask)
        self.log_image("prediction", self.model.predict(image + self.change))

        bar = trange(20_000)
        for self.step in bar:
            grad, value = self.grad_func(self.change, image=image, mask=mask)

            self.change -= self.lr * grad.detach()
            if self.step != 0 and self.step % 5_000:
                self.lr *= 0.1

            if self.step % 100 == 0:
                self.log_value("loss", value)
                self.log_value("lr", self.lr)
                if self.step % 1_000 == 0:
                    self.log_image("updated_image", (image + self.change)[0])
                    self.log_image("change", symetric_color_mapping(self.change))
                    self.log_image("prediction", self.model.predict(image + self.change))

            bar.set_description(f"loss: {round(value.detach().item(), 6)}, lr: {round(self.lr, 6)}")

            del grad, value

    def log_image(self, name: str, image: torch.Tensor) -> None:
        """
        Logs given images under the given dataset label.

        Args:
            name: name or title for image on tensorboard
            image: torch tensor with image data
        """
        # log in tensorboard
        self.writer.add_image(
                f"{name}",
                image[:, ::2, ::2],    # type: ignore
                global_step=self.step
            )  # type: ignore

        self.writer.flush()  # type: ignore

    def log_value(self, name: str, value: float) -> None:
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
                global_step=self.step
            )  # type: ignore

        self.writer.flush()  # type: ignore


def main():
    torch.manual_seed(42)

    model = SimpleUNet(in_channels=1)
    model.load()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    framework = Framework(model, optimizer, nn.CrossEntropyLoss(), (1, 64, 64), device)

    dataset = DummyDataset(100, (64, 64), artefact=True)
    image, mask = dataset[0]

    framework.process(image[None], mask[None])


if __name__ == '__main__':
    main()
