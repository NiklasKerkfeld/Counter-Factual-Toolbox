import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from dataset import DummyDataset
from model import SimpleUNet


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 trainloader: torch.utils.data.DataLoader,
                 testloader: torch.utils.data.DataLoader,
                 optimizer: optim.Optimizer,
                 loss_fn: nn.Module,
                 device: torch.device,
                 name: str = 'model'):

        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.name = name

    def train(self, epochs: int = 10):
        self.model.to(self.device)

        bar = trange(epochs)
        for _ in bar:
            train_loss = self.train_epoch()
            valid_loss = self.valid()

            bar.set_description(f"training loss: {train_loss}, validation loss: {valid_loss}")

        self.model.save()

    def train_epoch(self) -> float:
        self.model.train()

        losses = []

        for images, masks in self.trainloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, masks)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            del images, masks, loss

        return torch.mean(torch.tensor(losses)).item()

    def valid(self) -> float:
        self.model.eval()

        losses = []
        for images, masks in self.testloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.no_grad():
                output = self.model(images)
                loss = self.loss_fn(output, masks)
                losses.append(loss.item())

            del images, masks, loss

        return torch.mean(torch.tensor(losses)).item()


def main():
    torch.manual_seed(42)
    model = SimpleUNet(in_channels=1)

    train_dataset = DummyDataset(100, (64, 64), artefact=False)
    valid_dataset = DummyDataset(30, (64, 64), artefact=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    trainer = Trainer(model, train_loader, valid_loader, optimizer, loss_fn, device)
    trainer.train(epochs=10)

    example, target = valid_dataset[0]
    pred = model(example[None].to(device)).detach().cpu()[:, 1]

    os.makedirs("results", exist_ok=True)

    plt.title("Image")
    plt.imshow(example[0], cmap='gray')
    plt.savefig("results/image")

    plt.title("Prediction")
    plt.imshow(pred[0], cmap='gray')
    plt.savefig("results/prediction")


if __name__ == '__main__':
    main()
