from typing import Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from tqdm import trange

from src.Framework.Loss import Loss
from src.Framework.utils import normalize, change_visualization


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_shape: Tuple[int, int, int]):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.change = nn.Parameter(torch.zeros(input_shape))

    def forward(self, x):
        model_input = x + self.change
        x = self.model(model_input)
        return x, model_input

    def predict(self, image):
        with torch.no_grad():
            pred, _ = self(image)
            pred = torch.nn.functional.softmax(pred, dim=1)

        return pred[:, 1]


class Framework:
    def __init__(self, model: nn.Module,
                 input_shape: Tuple[int, int, int],
                 device: torch.device,
                 name: str = "framework",
                 lr: float = 1e-4,
                 ):

        self.model = ModelWrapper(model, input_shape)

        self.loss_fn = Loss(channel=input_shape[0])

        self.device = device
        self.name = name

        self.step = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20_000, gamma=0.1)

        # setup tensorboard
        train_log_dir = f"logs/Framework/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

    def process(self, image: torch.Tensor, mask: torch.Tensor) -> None:
        self.model.eval()

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        image_gpu = image.to(self.device)
        mask = mask.to(self.device)

        # logging 'adc', 'hbv', 't2w'
        self.log_mri("image", normalize(image_gpu))
        self.log_mri("update", normalize(image_gpu))
        self.log_image("target/mask", mask)
        self.log_image("target/init_prediction", self.model.predict(image_gpu + self.model.change))

        bar = trange(1, 10_001)
        for self.step in bar:
            self.optimizer.zero_grad()
            pred, model_input = self.model(image_gpu)
            loss = self.loss_fn(pred, model_input, mask, self.model.change)
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            loss = loss.detach().cpu().item()

            # logging
            if self.step == 1 or self.step % 100 == 0:
                self.log_value("loss", loss)
                self.log_value("lr", self.optimizer.param_groups[0]['lr'])
                if self.step % 2_000 == 0:
                    change = self.model.change.detach()
                    self.log_mri("update", normalize((image_gpu + change)))
                    self.log_image("update/change", change_visualization(change, normalize(image[0, 0])))
                    self.log_image("target/prediction", self.model.predict(image_gpu + change))

            bar.set_description(
                f"loss: {round(loss, 6)}, lr: {round(self.optimizer.param_groups[0]['lr'], 10)}")

            del loss

    def log_mri(self, name: str, image: torch.Tensor) -> None:
        """
        Logs given images under the given dataset label.

        Args:
            name: name or title for image on tensorboard
            image: torch tensor with image data
        """
        # log in tensorboard
        for channel in range(image.shape[1]):
            self.writer.add_image(
                f"{name}/channel{channel}",
                torch.rot90(image[0, channel, None], k=3, dims=(-2, -1)),  # type: ignore
                global_step=self.step
            )  # type: ignore

        self.writer.flush()  # type: ignore

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
            torch.rot90(image, k=3, dims=(-2, -1)),  # type: ignore
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
