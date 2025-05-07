import glob
import os.path
from copy import deepcopy

import monai.data
import numpy as np
import torch
from monai.transforms import SaveImage
from tqdm import tqdm

from src.Framework.utils import get_network, load_data


def save(image: torch.Tensor, path: str, example: monai.data.MetaTensor):
        save_image = SaveImage(output_dir=path, output_postfix='pred', output_dtype=np.int8)
        output_image = deepcopy(example)
        output_image.set_array(image)

        save_image(output_image)


def main(dataset_path: str):
    network = get_network(configuration='3d_fullres', fold=0)
    dataset = [x for x in glob.glob(f"{dataset_path}/sub*") if os.path.isdir(x)]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    network.to(device)

    for folder in tqdm(dataset, desc='predicting'):
        item = load_data(folder, device)
        with torch.no_grad():
            pass
            pred = network(item['tensor'])
            pred = torch.nn.functional.softmax(pred, dim=1)[:, 1] > .5

        save(pred, f"{folder}/anat", item['target'])


if __name__ == '__main__':
    main("data/Dataset101_fcd")
