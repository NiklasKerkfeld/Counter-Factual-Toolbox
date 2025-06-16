from typing import Tuple, Sequence

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from src.Architecture.Generator import Generator


class ScaleGenerator(Generator):
    def __init__(self, model: nn.Module, loss=CrossEntropyLoss(), alpha: float = 1.0):
        """
        Super class for image adaption.

        Args:
            model: model used for prediction
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, loss, alpha)

        self.scale = nn.Parameter(torch.tensor([1.0]))

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Args:
            input: input tensor

        Returns:
            the adapted image and the cost of that adaption.
        """
        input = input * self.scale

        return input, self.alpha * torch.abs(1 - self.scale)


class ShiftGenerator(Generator):
    def __init__(self, model: nn.Module, loss=CrossEntropyLoss(), alpha: float = 1.0):
        """
        Super class for image adaption.

        Args:
            model: model used for prediction
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, loss, alpha)

        self.shift = nn.Parameter(torch.tensor([0.0]))

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Args:
            input: input tensor

        Returns:
            the adapted image and the cost of that adaption.
        """
        input += self.shift

        return input, self.alpha * torch.abs(self.shift)


class ScaleAndShiftGenerator(Generator):
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 loss=CrossEntropyLoss(),
                 alpha: float = 1.0):
        """
        Super class for image adaption.

        Args:
            model: model used for prediction
            image_shape: shape of the input image (B, C, H, W)
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__(model, loss, alpha)

        self.scale = nn.Parameter(torch.ones((image_shape[0], image_shape[1], 1, 1)))
        self.shift = nn.Parameter(torch.zeros(image_shape[0], image_shape[1], 1, 1))

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Args:
            input: input tensor

        Returns:
            the adapted image and the cost of that adaption.
        """
        input = input * self.scale
        input = input + self.shift

        cost = torch.mean(torch.abs(1 - self.scale)) + torch.mean(torch.abs(self.shift))

        return input, self.alpha * cost

    def plot_visualization(self, image: torch.Tensor, new_image: torch.Tensor):
        print(f"{self.scale=}\n{self.shift=}")
