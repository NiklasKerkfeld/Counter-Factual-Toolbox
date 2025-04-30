import torch

from Framework.Framework import Framework
from src.Model.model import SimpleUNet
from src.Picai.utils import get_dataset, load_image
from src.fcd.utils import load_data, get_network


def fcd():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    item = load_data("nnUNet/nnUNet_raw/Dataset101_fcd/sub-00003", device=torch.device('cpu'), slice=161)
    print(f"{item['tensor'].shape=}")

    model = get_network(configuration='2d', fold=0)
    framework = Framework(model, item['tensor'].shape, device, name="fcd5")

    framework.process(item['tensor'][None], item['roi'].long())


def picai():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    model = SimpleUNet(in_channels=3)
    model.load('train10_es')

    framework = Framework(model, (3, 256, 256), device, name="mri45")

    item = load_image("data/all_lesions/valid/10121_1000121/20")

    framework.process(item['tensor'][None], item['lesion'][None].long())


if __name__ == '__main__':
    fcd()
