from typing import Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import trange

from dataset import DummyDataset
from model import SimpleUNet
from utils import symetric_color_mapping


class Loss(nn.Module):
    def __init__(self, loss_fn, beta: float = .5, gamma: float = .5):
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


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_shape: Tuple[int, int, int]):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.change = nn.Parameter(torch.zeros(input_shape))

    def forward(self, x):
        model_input = x + self.change
        x = self.model(model_input)
        return x, model_input

    def predict(self, image):
        with torch.no_grad():
            pred, _ = self(image)
            pred = torch.nn.functional.softmax(pred, dim=1)

        return pred[:, 0]


class Framework:
    def __init__(self, model: nn.Module,
                 loss_fn: nn.Module,
                 input_shape: Tuple[int, int, int],
                 device: torch.device,
                 name: str = "framework",
                 lr: float = 1e-4,
                 ):

        self.model = ModelWrapper(model, input_shape)

        self.loss_fn = Loss(loss_fn)

        self.device = device
        self.name = name

        self.step = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5000, gamma=0.5)

        # setup tensorboard
        train_log_dir = f"logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

    def process(self, image: torch.Tensor, mask: torch.Tensor) -> None:
        self.model.eval()

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        image = image.to(self.device)
        mask = mask.to(self.device)

        # logging
        self.log_image("image", image[0])
        self.log_image("mask", mask)
        self.log_image("prediction", self.model.predict(image + self.model.change))

        bar = trange(20_000)
        for self.step in bar:
            self.optimizer.zero_grad()
            pred, model_input = self.model(image)
            loss = self.loss_fn(pred, model_input, mask, self.model.change)
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            loss = loss.detach().cpu().item()

            # logging
            if self.step % 100 == 0:
                self.log_value("loss", loss)
                self.log_value("lr", self.optimizer.param_groups[0]['lr'])
                if self.step % 1_000 == 0:
                    change = self.model.change.detach()
                    self.log_image("updated_image", (image + change)[0])
                    self.log_image("change", symetric_color_mapping(change))
                    self.log_image("prediction", self.model.predict(image + change))

            bar.set_description(f"loss: {round(loss, 6)}, lr: {round(self.optimizer.param_groups[0]['lr'], 6)}")

            del loss

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
            image[:, ::2, ::2],  # type: ignore
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    framework = Framework(model, nn.CrossEntropyLoss(), (1, 64, 64), device, name="framework12")

    dataset = DummyDataset(100, (64, 64), artefact=True, reduction=True)
    image, mask = dataset[0]

    framework.process(image[None], mask[None])


if __name__ == '__main__':
    main()
