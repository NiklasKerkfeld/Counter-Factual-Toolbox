from typing import Tuple, List

import torch
from torch.func import grad_and_value
from torch import nn
from tqdm import trange

import matplotlib as mpl
from matplotlib import pyplot as plt

from dataset import DummyDataset
from model import SimpleUNet


class Loss(nn.Module):
    def __init__(self, loss_fn, beta: float = 1., gamma: float = 1.):
        super().__init__()
        self.loss_fn = loss_fn
        self.beta = beta
        self.gamma = gamma
        self.relu = nn.ReLU()

        self.grad_x = torch.tensor([[[[1., 0., -1.],
                                       [2., 0., -2.],
                                       [1., 0., -1.]]]], requires_grad=False)

        self.grad_y = torch.tensor([[[[1., 2., 1.],
                                       [0., 0., 0.],
                                       [-1., -2., -1.]]]], requires_grad=False)

    def magnitude(self, change):
        dx = torch.nn.functional.conv2d(change[None], self.grad_x)
        dy = torch.nn.functional.conv2d(change[None], self.grad_y)
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


class Framework:
    def __init__(self, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 input_shape: Tuple[int, int],
                 lr: float = 0.1,
                 decay: float = 1.0):

        self.model = model
        self.model.requires_grad = False

        self.optimizer = optimizer
        self.loss_fn = Loss(loss_fn)

        self.change = torch.zeros(input_shape)
        self.lr = lr
        self.decay = decay


    def process(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor, List[float]]:

        with torch.no_grad():
            pred_before = torch.nn.functional.softmax(self.model(image), dim=1)

        def func(change, image, mask):
            x = image + change
            pred = self.model(x)
            loss = self.loss_fn(pred, x, mask, change)
            return loss

        grad_func = grad_and_value(func)

        losses = []
        bar = trange(10_000)
        for _ in bar:
            grad, value = grad_func(self.change, image=image, mask=mask)

            self.change -= self.lr * grad.detach()
            if len(losses) == 5_000:
                self.lr *= 0.1

            bar.set_description(f"loss: {round(value.detach().item(), 6)}, lr: {round(self.lr, 6)}")
            losses.append(value.detach().item())

            del grad, value

        with torch.no_grad():
            pred_after = torch.nn.functional.softmax(self.model(image + self.change), dim=1)

        return self.change.detach(), pred_before[0, 1], pred_after[0, 1], losses


def plot(image, mask, change, pred_before, pred_after, loss_curve):
    centered_norm = mpl.colors.CenteredNorm()
    # norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=False)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 0.05, 1, 0.05])

    axs = [[None for _ in range(5)] for _ in range(3)]

    axs[0][0] = fig.add_subplot(gs[0, 0])
    axs[0][0].set_title("image")
    axs[0][0].imshow(image[0], cmap='gray')
    axs[0][0].axis('off')

    axs[0][1] = fig.add_subplot(gs[0, 1])
    axs[0][1].set_title("mask")
    axs[0][1].imshow(mask, cmap='gray')
    axs[0][1].axis('off')

    axs[0][3] = fig.add_subplot(gs[0, 3])
    axs[0][3].set_title("pred before")
    axs[0][3].imshow(pred_before, cmap='gray')
    axs[0][3].axis('off')

    axs[1][0] = fig.add_subplot(gs[1, 0])
    axs[1][0].set_title("changed image")
    axs[1][0].imshow(image[0] + change[0], cmap='gray')
    axs[1][0].axis('off')

    axs[1][1] = fig.add_subplot(gs[1, 1])
    axs[1][1].set_title("change")
    im_change = axs[1][1].imshow(change[0], norm=centered_norm, cmap='bwr')
    axs[1][1].axis('off')
    fig.colorbar(im_change, cax=fig.add_subplot(gs[1, 2]))

    axs[1][3] = fig.add_subplot(gs[1, 3])
    axs[1][3].set_title("pred after")
    axs[1][3].imshow(pred_after, cmap='gray')
    axs[1][3].axis('off')

    # Create one large subplot spanning (2, 0) and (2, 1)
    ax_loss = fig.add_subplot(gs[2, 0:3])
    ax_loss.set_title("Loss Curve")
    ax_loss.plot(loss_curve, color='blue')
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")

    axs[2][3] = fig.add_subplot(gs[2, 3])
    axs[2][3].set_title("difference pred")
    im_diff = axs[2][3].imshow(pred_after - pred_before, norm=centered_norm, cmap='bwr')
    axs[2][3].axis('off')
    fig.colorbar(im_diff, ax=axs[2][3])

    plt.tight_layout()
    plt.show()



def main():
    torch.manual_seed(42)

    model = SimpleUNet(in_channels=1)
    model.load()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    framework = Framework(model, optimizer, nn.CrossEntropyLoss(), (1, 64, 64))

    dataset = DummyDataset(100, (64, 64), artefact=True)
    image, mask = dataset[0]

    change, pred_before, pred_after, losses = framework.process(image[None], mask[None])

    print("change: ", change.max(), change.min(), change.mean(), change.std(), change.median())
    new_image = image + change
    print("new_image: ", new_image.min(), new_image.max(), new_image.mean(), new_image.std(), new_image.median())

    plot(image, mask, change, pred_before, pred_after, losses)


if __name__ == '__main__':
    main()
