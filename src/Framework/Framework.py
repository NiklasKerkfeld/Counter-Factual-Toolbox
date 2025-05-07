from typing import Tuple, Optional

import torch
from monai.metrics import DiceMetric
from torch import nn
from tqdm import trange

from src.Framework.Loss import Loss
from src.Framework.config import STEPS, LR
from src.Visualization.Logger import Logger


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
                 logger: Logger,
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

        self.metric = DiceMetric(include_background=False, num_classes=1)
        self.logger = logger
        self.step = 0
        self.num_steps = steps

    def process(self, image: torch.Tensor, mask: torch.Tensor) -> None:
        self.model.eval()

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        image_gpu = image.to(self.device)
        mask = mask.to(self.device)

        bar = trange(1, self.num_steps + 1)
        for self.step in bar:
            # process
            self.optimizer.zero_grad()
            pred, model_input = self.model(image_gpu)
            loss, loss_dict = self.loss_fn(pred, mask, self.model.change, model_input)
            loss.backward()
            self.optimizer.step()

            # logging
            if self.step == 1 or self.step % 10 == 0:
                pred = torch.argmax(pred, dim=1, keepdim=True)
                self.metric(pred, mask[None])
                loss_dict['dice'] = self.metric.aggregate().item()
                self.metric.reset()

                print(f"{self.step=}")
                print(loss_dict)
                print()

                self.logger.log_values(self.step,
                                       **loss_dict,
                                       lr=self.optimizer.param_groups[0]['lr'])

                change = self.model.change.detach()
                self.logger.log_change(self.step, change)
                self.logger.log_prediction(self.step, self.model.predict(image_gpu + change))

            bar.set_description(
                f"loss: {round(loss.detach().cpu().item(), 6)}, lr: {round(self.optimizer.param_groups[0]['lr'], 10)}")
