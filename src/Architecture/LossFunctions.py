import torch
from torch import nn
from torch.nn import functional as F


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = torch.argmax(F.softmax(input, dim=1), dim=1)
            mask = pred != target

        loss_map = F.cross_entropy(input, target, reduction='none')

        return torch.mean(loss_map * mask)
