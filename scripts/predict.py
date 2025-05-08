import glob
import os.path

import torch
from tqdm import tqdm

from src.Framework.utils import get_network, load_data, save


def main(dataset_path: str):
    network = get_network(configuration='3d_fullres', fold=0)
    dataset = [x for x in glob.glob(f"{dataset_path}/sub*") if os.path.isdir(x)]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    network.to(device)

    for folder in tqdm(dataset, desc='predicting'):
        item = load_data(folder, device)
        with torch.no_grad():
            pass
            pred = network(item['tensor'][None])
            pred = torch.nn.functional.softmax(pred, dim=1)[:, 1] > .5

        save(pred, f"{folder}/anat", item['target'])


if __name__ == '__main__':
    main("data/Dataset101_fcd")
