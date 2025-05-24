import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Adversarial.Dataset2D import Dataset2D
from src.Architecture.AdversarialGenerator import AdversarialGenerator
from src.Framework.utils import get_network


class Trainer:
    def __init__(self):
        self.iterations = 10
        self.epochs = 5
        self.steps = 10

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = get_network(configuration='2d', fold=0)
        self.generator = AdversarialGenerator(self.model, (2, 256, 256))
        self.generator.to(self.device)

        dataset = Dataset2D("data/Dataset101_fcd")
        self.dataloader_gen = DataLoader(dataset)
        self.dataloader_train = DataLoader(dataset,
                                           batch_size=64,
                                           shuffle=True)

    def train_adversarial(self):
        optimizer = torch.optim.Adam(self.generator.adversarial.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        for e in range(self.epochs):
            print(f"start epoch {e+1}:")
            loss_lst = []
            bar = tqdm(self.dataloader_train)
            for image, _, change in bar:
                image = image.to(self.device)
                change = change.to(self.device)

                optimizer.zero_grad()
                pred = self.generator.adversarial(image)
                loss = loss_fn(pred, change)
                loss.backward()
                optimizer.step()

                loss_lst.append(loss.detach().cpu().item())
                bar.set_description(f"training loss: {loss.item():.8f}")

            print(f"finished training epoch {e+1} with an average loss of {np.mean(loss_lst)}")

    def generate(self, alpha: float = 1.0):
        self.generator.alpha = alpha
        loss_lst = []

        bar = tqdm(self.dataloader_gen, desc='generating')
        for image, target, change in bar:
            self.generator.reset()
            optimizer = torch.optim.Adam([self.generator.change], lr=1e-1)

            image = image.to(self.device)
            target = target.to(self.device)

            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = self.generator(image, target)
                loss.backward()
                optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            change.copy_(self.generator.change.data.cpu())

        print(f"finished generating with an average loss of {np.mean(loss_lst)}")

    def train(self):
        for i in range(self.iterations):
            self.generate(alpha=min([float(i), 1.0]))
            self.train_adversarial()


if __name__ == '__main__':
    Trainer().train()
