import torch
from monai.data import DataLoader

from torch import nn, optim

from src.Model.Trainer import Trainer
from src.Model.model import SimpleUNet
from src.Picai.utils import get_dataset


def main():
    torch.manual_seed(42)
    model = SimpleUNet(in_channels=3)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    train_dataset = get_dataset("data/preprocessed/train", train_mode=True, device=device)
    valid_dataset = get_dataset("data/preprocessed/valid", train_mode=False, device=device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, valid_loader, optimizer, loss_fn, torch.device(device), name='train6')
    trainer.train(epochs=25)


if __name__ == '__main__':
    main()
