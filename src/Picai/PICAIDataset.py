import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class PicaiDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.folder = glob.glob(f"{path}/*/*")

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        input_data = torch.from_numpy(np.load(f"{self.folder[idx]}/input.npy"))
        target_data = torch.from_numpy(np.load(f"{self.folder[idx]}/target.npy")[0])

        return input_data.float(), target_data.long()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os

    dataset = PicaiDataset("data/preprocessed/train")
    print(len(dataset))

    input_data, target_data = dataset[0]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)

    axs = [[None for _ in range(2)] for _ in range(2)]

    axs[0][0] = fig.add_subplot(gs[0, 0])
    axs[0][0].set_title("t2w")
    axs[0][0].imshow(input_data[0], cmap='gray')
    axs[0][0].axis('off')

    axs[0][1] = fig.add_subplot(gs[0, 1])
    axs[0][1].set_title("hbv")
    axs[0][1].imshow(input_data[1], cmap='gray')
    axs[0][1].axis('off')

    axs[1][0] = fig.add_subplot(gs[1, 0])
    axs[1][0].set_title("adc")
    axs[1][0].imshow(input_data[2], cmap='gray')
    axs[1][0].axis('off')

    axs[1][1] = fig.add_subplot(gs[1, 1])
    axs[1][1].set_title("lesion")
    axs[1][1].imshow(target_data[0], cmap='gray')
    axs[1][1].axis('off')

    plt.tight_layout()

    os.makedirs("../../results", exist_ok=True)
    plt.savefig("results/example.png")
