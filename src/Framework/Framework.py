from typing import Tuple, Optional

import torch
from torch import nn
from tqdm import trange

from src.Framework.Loss import Loss
from src.Framework.config import STEPS, LR
from src.Framework.utils import dice
from src.Visualization.Logger import Logger


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_shape: Tuple[int, int, int]):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.change = nn.Parameter(torch.zeros(input_shape))

    def input_image(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.change

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model_input = self.input_image(x)
        x = self.model(model_input)
        return x

    def predict(self, image):
        with torch.no_grad():
            pred, _ = self(image)
            pred = torch.nn.functional.softmax(pred, dim=1)

        return pred[:, 1]


class Framework:
    def __init__(self, model: nn.Module,
                 input_shape: Tuple[int, int, int],
                 device: Optional[torch.device] = None,
                 lr: float = LR,
                 steps: int = STEPS
                 ):

        self.model = ModelWrapper(model, input_shape)
        self.loss_fn = Loss(channel=input_shape[0])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"device: {self.device}\n")

        self.logger = None
        self.step = 0
        self.num_steps = steps

    def process(self, image: torch.Tensor, target: torch.Tensor, logger: Logger) -> None:
        self.logger = logger

        self.model.model.eval()

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        image_gpu = image.to(self.device)
        target = target.to(self.device)

        bar = trange(self.num_steps + 1)
        for self.step in bar:
            # logging change before backprob
            if self.step == 1 or self.step % 10 == 0:
                self.logger.log_change(self.step, self.model.change.detach())

            # process
            self.optimizer.zero_grad()
            pred = self.model(image_gpu)
            loss, loss_dict = self.loss_fn(pred, target, self.model.change)
            loss.backward()
            self.optimizer.step()

            # logging
            if self.step == 1 or self.step % 10 == 0:
                pred = torch.argmax(pred, dim=1)
                loss_dict['dice'] = dice(pred, target)

                self.logger.log_values(self.step,
                                       **loss_dict,
                                       lr=self.optimizer.param_groups[0]['lr'])

                self.logger.log_prediction(self.step, pred)

            bar.set_description(
                f"loss: {round(loss.detach().cpu().item(), 6)}, lr: {round(self.optimizer.param_groups[0]['lr'], 10)}")


    def generate(self, image: torch.Tensor, target: torch.Tensor):
        self.model.model.eval()

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        image_gpu = image.to(self.device)
        target = target.to(self.device)

        for self.step in range(self.num_steps + 1):
            # process
            self.optimizer.zero_grad()
            pred = self.model(image_gpu)
            loss, loss_dict = self.loss_fn(pred, target, self.model.change)
            loss.backward()
            self.optimizer.step()

        return self.model.change.detach().cpu()

