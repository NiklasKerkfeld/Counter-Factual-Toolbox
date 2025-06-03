from typing import List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from src.Architecture.Generator import Generator


class ComposeGenerator(Generator):
    def __init__(self, model: nn.Module, loss=CrossEntropyLoss(), alpha: float = 1.0, composion: List[Generator] = []):
        super(ComposeGenerator, self).__init__( model, loss, alpha)

        self.composion = composion

    def adapt(self, image: torch.Tensor):
        overall_cost = torch.tensor([0.0], device=image.device)
        for g in self.composion:
            image, cost = g.adapt(image)
            overall_cost += cost

        return image, overall_cost