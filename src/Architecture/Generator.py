"""Super class for image adaption."""

from typing import Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class Generator(nn.Module):
    """Super class for image adaption."""
    def __init__(self, model: nn.Module, alpha: float = 1.0):
        """
        Super class for image adaption.

        Args:
            model: model used for prediction
            alpha: weight of the adaption cost in comparison to the prediction loss
        """
        super().__init__()
        self.model = model
        self.alpha = alpha

        self.loss = CrossEntropyLoss()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward path for generation.

        Adopts image with adopt function. Then calculates prediction with loss and add them with the
        cost of the adoption.

        Args:
            input: input tensor
            target: target tensor

        Returns:
            over all loss
        """
        image, cost = self.adapt(input)
        output = self.model(image)
        loss = self.loss(output, target)
        return loss + self.alpha * cost

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapts the image and returns it together with the cost of the adaption.

        Args:
            input: input tensor

        Returns:
            the adapted image and the cost of that adaption.
        """
        return input, torch.tensor(0.0).to(input.device)