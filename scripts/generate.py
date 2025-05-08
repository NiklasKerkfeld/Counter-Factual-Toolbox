import glob
import os.path

import numpy as np
import torch
from tqdm import tqdm

from src.Framework.Framework import ModelWrapper, Framework
from src.Framework.utils import get_network, load_data, save, get_image_files, load_item


def main(dataset_path: str, output_path: str):
    network = ModelWrapper(get_network(configuration='3d_fullres', fold=0), (160, 256, 256))
    dataset = [x for x in glob.glob(f"{dataset_path}/sub*") if os.path.isdir(x)]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    network.to(device)

    for folder in tqdm(dataset, desc='generating'):
        item = load_data(folder, device=device)

        model = get_network(configuration='3d_fullres', fold=0)
        framework = Framework(model, item['tensor'].shape, device=device)
        change = framework.generate(item['tensor'][None], item['target'].long())

        save(change, f"{output_path}/{folder}", item['target'], post_fix='change', dtype=np.float32)


if __name__ == '__main__':
    main("data/Dataset101_fcd", "data/change")
