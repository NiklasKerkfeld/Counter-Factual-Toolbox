import torch

from Framework.Framework import Framework
from Model.model import SimpleUNet
from src.Picai.utils import get_dataset


def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    model = SimpleUNet(in_channels=3)
    model.load('train6_es')

    framework = Framework(model, (3, 256, 256), device, name="mri10")

    dataset = get_dataset("data/preprocessed/valid", train_mode=False, device='cpu')
    item = dataset[0]

    framework.process(item['tensor'][None], item['lesion'][None].long())


if __name__ == '__main__':
    main()
