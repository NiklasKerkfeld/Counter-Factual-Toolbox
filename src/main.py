import torch

from Framework.Framework import Framework
from Picai.PICAIDataset import PicaiDataset
from Model.model import SimpleUNet


def main():
    torch.manual_seed(42)

    model = SimpleUNet(in_channels=3)
    model.load()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    framework = Framework(model, (3, 256, 256), device, name="mri1")

    dataset = PicaiDataset("data/preprocessed/valid")
    image, mask = dataset[0]

    framework.process(image[None], mask[None])


if __name__ == '__main__':
    main()
