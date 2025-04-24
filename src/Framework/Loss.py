import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self,
                 weight_l1: float = 2.0,
                 weight_l2: float = 0.0,
                 weight_magnitude: float = 5.0,
                 weight_bi_model: float = 0.0,
                 weight_try: float = 0.0,
                 channel: int = 1):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_l2 = weight_l2
        self.weight_magnitude = weight_magnitude
        self.weight_bi_modal = weight_bi_model
        self.weight_try = weight_try
        self.channel = channel

        self.loss_fn = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()

        self.grad_x = nn.Parameter(torch.tensor([[[[1., -1.]]]]).repeat(self.channel, 1, 1, 1),
                                   requires_grad=False)
        self.grad_y = nn.Parameter(torch.tensor([[[[1.], [-1.]]]]).repeat(self.channel, 1, 1, 1),
                                   requires_grad=False)

    def forward(self, pred, input, target, change):
        # normal loss
        loss = self.loss_fn(pred, target)

        # penalize values out of image range
        loss += torch.sum(self.relu(input - 1))
        loss += torch.sum(self.relu(-input))

        # regularize
        loss += self.l1_regularization(self.weight_l1, change)
        loss += self.l2_regularization(self.weight_l2, change)
        loss += self.magnitude(self.weight_magnitude, change)
        loss += self.bi_modal_regularization(self.weight_bi_modal, change)
        loss += self.try_regularization(self.weight_try, change)

        return loss

    def l1_regularization(self, weight: float, change: torch.Tensor) -> torch.Tensor:
        if weight == 0.0:
            return torch.tensor(0.0)

        return weight * torch.mean(torch.abs(change))

    def l2_regularization(self, weight: float, change: torch.Tensor) -> torch.Tensor:
        if weight == 0.0:
            return torch.tensor(0.0)

        return weight * torch.mean(change ** 2)

    def magnitude(self, weight: float, change: torch.Tensor) -> torch.Tensor:
        if weight == 0.0:
            return torch.tensor(0.0)

        dx = torch.conv2d(change[None], self.grad_x, groups=self.channel)
        dy = torch.conv2d(change[None], self.grad_y, groups=self.channel)
        return weight * (torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy)))

    def bi_modal_regularization(self, weight: float, change: torch.Tensor) -> torch.Tensor:
        if weight == 0.0:
            return torch.tensor(0.0)

        a = torch.abs(change) * (1 - torch.abs(change))
        return weight * torch.mean(a)

    def try_regularization(self, weight: float, change: torch.Tensor) -> torch.Tensor:
        if weight == 0.0:
            return torch.tensor(0.0)

        max_values = torch.max_pool3d(change[None], kernel_size=(1, 3, 3), stride=1)
        min_values = -torch.max_pool3d(-change[None], kernel_size=(1, 3, 3), stride=1)
        difference = max_values - min_values

        return weight * torch.mean(difference)
