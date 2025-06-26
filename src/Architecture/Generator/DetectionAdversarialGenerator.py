from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.Architecture.Generator import AdversarialGenerator
from src.Architecture.LossFunctions import MaskedCrossEntropyLoss


class DetectionAdversarialGenerator(AdversarialGenerator):
    def __init__(self, model: nn.Module,
                 image: torch.tensor,
                 target: torch.tensor,
                 name: str = "AdversarialGenerator",
                 loss: nn.Module = MaskedCrossEntropyLoss(),
                 alpha: float = 1.0,
                 threshold: float = 1e-9):
        super().__init__(model, image, target, name, loss, alpha)
        self.threshold = threshold

    def adapt(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # calc updated image
        new_input = self.image + self.change
        mask = self.detection_mask(new_input)
        cost = torch.abs(self.change * mask).sum()

        self.mean_changes.append(torch.abs(self.parameter).mean().detach().cpu())
        return new_input, cost

    def detection_mask(self, new_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = self.adversarial(new_input)
            # pred *= pred.abs() > self.threshold
            pred = torch.sign(pred)
            mask = F.relu(pred * self.change)

        return mask