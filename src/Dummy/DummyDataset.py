from typing import Tuple

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, N,
                 shape: Tuple[int, int] = (100, 100),
                 noise: float = 0.1,
                 min_size: int = 10,
                 artefact: bool = False,
                 min_size_artefact: int = 5,
                 reduction: bool = False,
                 min_size_reduction: int = 5):
        self.N = N
        self.shape = shape
        self.noise = noise
        self.min_size = min_size
        self.artefact = artefact
        self.min_size_artefact = min_size_artefact
        self.reduction = reduction
        self.min_size_reduction = min_size_reduction

    def __getitem__(self, index):
        mask = torch.zeros((1, *self.shape), dtype=torch.float32)
        x1 = torch.randint(0, self.shape[0] - self.min_size, (1,)).item()
        x2 = torch.randint(x1 + self.min_size, self.shape[0], (1,)).item()

        y1 = torch.randint(0, self.shape[1] - self.min_size, (1,)).item()
        y2 = torch.randint(y1 + self.min_size, self.shape[1], (1,)).item()

        mask[:, x1:x2, y1:y2] = 1.0
        image = mask.clone()

        if self.reduction:
            x1 = torch.randint(x1, x2 - self.min_size_reduction, (1,)).item()
            x2 = torch.randint(x1 + self.min_size_reduction, x2, (1,)).item()

            y1 = torch.randint(y1, y2 - self.min_size_reduction, (1,)).item()
            y2 = torch.randint(y1 + self.min_size_reduction, y2, (1,)).item()

            image[:, x1:x2, y1:y2] = 0.0

        if self.artefact:
            x1 = torch.randint(0, self.shape[0] - self.min_size_artefact, (1,)).item()
            x2 = torch.randint(x1 + self.min_size_artefact, self.shape[0], (1,)).item()

            y1 = torch.randint(0, self.shape[1] - self.min_size_artefact, (1,)).item()
            y2 = torch.randint(y1 + self.min_size_artefact, self.shape[1], (1,)).item()

            image[:, x1:x2, y1:y2] = .5

        image = (image + (torch.randn(mask.shape) * self.noise)).clip(0., 1.)

        return image, mask.squeeze().long()

    def __len__(self):
        return self.N


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = DummyDataset(100, artefact=True, reduction=True)
    image, mask = dataset[0]

    print(image.shape, mask.shape)
    print(image.max(), image.min())
    print(mask.max(), mask.min())

    plt.title("Image")
    plt.imshow(image[0], cmap='gray')
    plt.savefig("results/image")

    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.savefig("results/mask")
