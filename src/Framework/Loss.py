import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self, beta: float = .0, gamma: float = 1.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.beta = beta
        self.gamma = gamma
        self.relu = nn.ReLU()

        self.grad_x = torch.tensor([[[[1., -1.],
                                      [1., -1.]]]], requires_grad=False).repeat(3, 1, 1, 1)
        self.grad_y = torch.tensor([[[[1., 1.],
                                      [-1., -1.]]]], requires_grad=False).repeat(3, 1, 1, 1)

    def to(self, device: torch.device):
        self.grad_x = self.grad_x.to(device)
        self.grad_y = self.grad_y.to(device)

    def magnitude(self, change):
        dx = torch.conv2d(change[None], self.grad_x, groups=3)
        dy = torch.conv2d(change[None], self.grad_y, groups=3)
        return torch.mean((torch.abs(dx) + torch.abs(dy)))

    def forward(self, pred, input, target, change):
        # normal loss
        loss = self.loss_fn(pred, target)

        # regularize to achieve small changes
        reg = torch.mean(torch.abs(change))

        # penalize values out of image range
        over = torch.sum(self.relu(input - 1))
        under = torch.sum(self.relu(-input))

        # penalize different neighbors
        magnitude = self.magnitude(change)

        return loss + self.beta * reg + over + under + self.gamma * magnitude