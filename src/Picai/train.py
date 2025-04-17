import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.Picai.PICAIDataset import PicaiDataset
from src.Model.Trainer import Trainer
from src.Model.model import SimpleUNet


def main():
    torch.manual_seed(42)
    model = SimpleUNet(in_channels=3)

    train_dataset = PicaiDataset("data/preprocessed/train")
    valid_dataset = PicaiDataset("data/preprocessed/valid")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    trainer = Trainer(model, train_loader, valid_loader, optimizer, loss_fn, device, name='train1')
    trainer.train(epochs=10)


if __name__ == '__main__':
    main()
