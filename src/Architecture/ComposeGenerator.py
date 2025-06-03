from typing import List, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from src.Architecture.Generator import Generator


class ComposeGenerator(Generator):
    def __init__(self, model: nn.Module, loss=CrossEntropyLoss(), alpha: float = 1.0,
                 composition: List[Generator] = [],
                 weights: Optional[List[float]] = None):
        super(ComposeGenerator, self).__init__( model, loss, alpha)

        self.composition = composition
        self.weights = weights if weights is not None else [1.0] * len(composition)

        if len(self.composition) != len(self.weights):
            raise ValueError("The number of weights and composition must be the same")

    def adapt(self, image: torch.Tensor):
        overall_cost = torch.tensor([0.0], device=image.device)
        for g, weight in zip(self.composition, self.weights):
            image, cost = g.adapt(image)
            overall_cost += weight * cost

        return image, overall_cost