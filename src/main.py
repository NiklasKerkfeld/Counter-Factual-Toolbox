import argparse

from src.Framework.Framework import Framework
from src.Framework.utils import get_network, get_image_files, load_item
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
    print(f"{item['target'].shape=}")

    model = get_network(configuration='3d_fullres', fold=0)
    framework = Framework(model, item['tensor'].shape)
    framework.process(item['tensor'][None], item['target'].long(), logger)

def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="path to image folder")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="name of this run in the log files."
    )

    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="path to image folder."
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(f"{args=}")

    main(name=args.name,
         data_path=args.path)
