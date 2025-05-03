import torch

from Framework.Framework import Framework
from src.Model.model import SimpleUNet
from src.Picai.utils import load_image
from src.fcd.utils import get_network, get_image_files, load_item
from src.Visualization.Logger import Logger


def main(name: str, data_path: str):
    item = get_image_files(data_path)
    # setup logger
    train_log_dir = f"logs/Logger/{name}"
    print(f"{train_log_dir=}")
    logger = Logger(train_log_dir,
                    images_paths={key: value for key, value in item.items() if key != 'target'},
                    target_path=item['target'])  # type: ignore

    item = load_item(item)
    print(f"{item['tensor'].shape=}")

    model = get_network(configuration='3d_fullres', fold=0)
    framework = Framework(model, item['tensor'].shape, logger)
    framework.process(item['tensor'][None], item['target'].long())


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
    main(name='run1',
         data_path="nnUNet/nnUNet_raw/Dataset101_fcd/sub-00003")
