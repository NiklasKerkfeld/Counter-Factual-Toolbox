import os

import torch
from matplotlib import pyplot as plt

from src.Picai.PICAIDataset import PicaiDataset
from src.Model.model import SimpleUNet


def main():
    model = SimpleUNet(in_channels=3)
    model.load()

    dataset = PicaiDataset("data/preprocessed/valid")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    input_data, target = dataset[0]

    input_data = input_data.to(device)
    model.to(device)

    with torch.no_grad():
        pred = model(input_data[None])
        pred = torch.nn.functional.softmax(pred, dim=1)[0, 1].cpu()

    input_data = input_data.cpu()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    axs = [[None for _ in range(3)] for _ in range(2)]

    axs[0][0] = fig.add_subplot(gs[0, 0])
    axs[0][0].set_title("t2w")
    axs[0][0].imshow(input_data[0], cmap='gray')
    axs[0][0].axis('off')

    axs[0][1] = fig.add_subplot(gs[0, 1])
    axs[0][1].set_title("hbv")
    axs[0][1].imshow(input_data[1], cmap='gray')
    axs[0][1].axis('off')

    axs[0][2] = fig.add_subplot(gs[0, 2])
    axs[0][2].set_title("adc")
    axs[0][2].imshow(input_data[2], cmap='gray')
    axs[0][2].axis('off')

    axs[1][0] = fig.add_subplot(gs[1, 0])
    axs[1][0].set_title("target")
    axs[1][0].imshow(target, cmap='gray')
    axs[1][0].axis('off')

    axs[1][1] = fig.add_subplot(gs[1, 1])
    axs[1][1].set_title("prediction")
    axs[1][1].imshow(pred, cmap='gray')
    axs[1][1].axis('off')

    plt.tight_layout()

    os.makedirs("../../results", exist_ok=True)
    plt.savefig("results/prediction.png")


if __name__ == '__main__':
    main()
