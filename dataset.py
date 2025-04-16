from typing import Tuple

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, N,
                 shape: Tuple[int, int] = (100, 100),
                 noise: float = 0.3,
                 min_size: int = 5,
                 artefact: bool = False):
        self.N = N
        self.shape = shape
        self.noise = noise
        self.min_size = min_size
        self.artefact = artefact

    def __getitem__(self, index):
        mask = torch.zeros((1, *self.shape), dtype=torch.float32)
        x1 = torch.randint(0, self.shape[0] - self.min_size, (1,)).item()
        x2 = torch.randint(x1 + self.min_size, self.shape[0], (1,)).item()

        y1 = torch.randint(0, self.shape[1] - self.min_size, (1,)).item()
        y2 = torch.randint(y1 + self.min_size, self.shape[1], (1,)).item()

        mask[:, x1:x2, y1:y2] = 1.0
        image = mask.clone()

        if self.artefact:
            x1 = torch.randint(0, self.shape[0] - self.min_size, (1,)).item()
            x2 = torch.randint(x1 + self.min_size, self.shape[0], (1,)).item()

            y1 = torch.randint(0, self.shape[1] - self.min_size, (1,)).item()
            y2 = torch.randint(y1 + self.min_size, self.shape[1], (1,)).item()

            image[:, x1:x2, y1:y2] = .5

        image = (image + (torch.randn(mask.shape) * self.noise)).clip(0., 1.)

        return image, mask.squeeze().long()

    def __len__(self):
        return self.N


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = DummyDataset(100)
    image, mask = dataset[0]

    print(image.shape, mask.shape)
    print(image.max(), image.min())
    print(mask.max(), mask.min())

    plt.title("Image")
    plt.imshow(image[0], cmap='gray')
    plt.show()

    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.show()
