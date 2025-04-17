import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.Dummy.DummyDataset import DummyDataset
from Trainer import Trainer
from src.Model.model import SimpleUNet


def main():
    torch.manual_seed(42)
    model = SimpleUNet(in_channels=1)

    train_dataset = DummyDataset(100, (128, 128), artefact=False)
    valid_dataset = DummyDataset(30, (128, 128), artefact=False)
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

    os.makedirs("../../results", exist_ok=True)

    plt.title("Image")
    plt.imshow(example[0], cmap='gray')
    plt.savefig("results/image")

    plt.title("Prediction")
    plt.imshow(pred[0], cmap='gray')
    plt.savefig("results/prediction")


if __name__ == '__main__':
    main()
