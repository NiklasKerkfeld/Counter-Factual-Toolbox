from typing import Sequence, Tuple

import torch
from monai.networks.nets import BasicUNet
from torch import nn

from src.Architecture.Generator import Generator


class AdversarialGenerator(Generator):
    def __init__(self, model: nn.Module,
                 image_shape: Sequence[int],
                 alpha: float = 1.0):
        super().__init__(model, alpha)
        self.image_shape = image_shape

        self.adversarial = BasicUNet(in_channels=2,
                                     out_channels=1,
                                     spatial_dims=2,
                                     features=(64, 128, 256, 512, 1024, 128))

        self.change = nn.Parameter(torch.zeros(*image_shape))

    def image(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input =  input + self.change

        # cost are the predicted change by the adversarial
        cost = torch.mean(self.adversarial(new_input))

        return new_input, cost
