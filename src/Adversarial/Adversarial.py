import torch
from torch import nn
from tqdm import trange, tqdm

from src.Architecture.AdversarialGenerator import AdversarialGenerator
from src.Framework.utils import get_network

class Trainer:
    def __init__(self):
        self.model = get_network(configuration='2d', fold=0)
        self.generator = AdversarialGenerator(self.model, (2, 160, 256))


    def train_adversarial(self, epochs: int = 5):
        model = self.generator.adversarial
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()

        for _ in trange(epochs):
            for image, target in bar := tqdm(self.train_dataloader):
                optimizer.zero_grad()
                pred = model(image)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()

                bar.set_description(f"loss: {loss.item():.5f}")



    def generate(self):
        pass


    def train(self, iterations: int = 10):

        for i in range(iterations):
            self.train_adversarial()
            self.generate()

    
   
if __name__ == '__main__':
    Trainer().train()
