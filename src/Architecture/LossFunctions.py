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


class DistanceLoss(nn.Module):
    def __init__(self, width: float):
        super().__init__()
        self.width = width ** 2

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        attention = torch.abs(tensor_a) + torch.abs(tensor_b)
        difference = torch.nn.functional.relu(self.width - torch.abs(tensor_a - tensor_b))
        return torch.mean(difference * attention)


class RelativeL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1.0))


if __name__ == '__main__':
    pred = torch.tensor([0.01])
    target = torch.tensor([0.0])

    rel_l1 = RelativeL1Loss()
    loss = rel_l1(pred, target)
    print(loss)