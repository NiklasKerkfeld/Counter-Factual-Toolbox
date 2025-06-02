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
                                     out_channels=2,
                                     spatial_dims=2,
                                     features=(64, 128, 256, 512, 1024, 128))

        self.change = nn.Parameter(torch.zeros(*self.image_shape))

    def adapt(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input = input + self.change

        # cost are the predicted change by the adversarial
        if self.alpha != 0.0:
            cost = torch.mean(torch.abs(self.adversarial(new_input)))
        else:
            cost = torch.tensor(0.0, device=input.device)

        return new_input, cost

    def reset(self):
        device = self.change.device
        self.change = nn.Parameter(torch.zeros(*self.image_shape, device=device))


if __name__ == '__main__':
    from src.utils import get_network

    model = get_network(configuration='2d', fold=0)
    generator = AdversarialGenerator(model, (256, 256))

    print(generator)
