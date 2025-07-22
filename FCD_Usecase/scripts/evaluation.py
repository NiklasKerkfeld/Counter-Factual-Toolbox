import argparse
import csv
import glob
import os
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from CounterFactualToolbox.Generator import (SmoothChangeGenerator,
                                             DeformationGenerator,
                                             RegularizedChangeGenerator,
                                             AdversarialGenerator,
                                             DetectionAdversarialGenerator)
from FCD_Usecase.scripts.utils.utils import get_network, load_image, intersection_over_union

exceptions = ['sub-00002',
              'sub-00074',
              'sub-00130',
              'sub-00120',
              'sub-00027',
              'sub-00018',
              'sub-00053',
              'sub-00112']


class Dataset2D(Dataset):
    def __init__(self, path: str, slice_dim: int = 2):
        super().__init__()
        self.slice_dim = slice_dim + 1

        self.data = {}
        self.len = 0

        for x in tqdm([x for x in sorted(glob.glob(f"{path}/sub-*"), key=lambda x: int(x[-5:])) if
                       os.path.isdir(x)],
                      desc='loading data'):
            patient = os.path.basename(x)
            if patient in exceptions:
                continue

            item, num_slices = self.get_image(x)

            for i in range(num_slices):
                self.data[self.len] = (patient, item, i)
                self.len += 1

    def get_image(self, path: str):
        item = load_image(path)

        num_slices = item['tensor'].shape[self.slice_dim]

        return item, num_slices

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        patient, item, i = self.data[index]
        image = item['tensor'].select(self.slice_dim, i)
        target = item['target'].select(self.slice_dim, i)

        return patient, i, image, target[0]


def generate(method: str, model: nn.Module, image: torch.Tensor, target: torch.Tensor, device: torch.device):
    prediction = model(image)[:, 1] > .5

    if prediction.sum() == 0.0 and target.sum() == 0.0:
        return None, None, None

    if method == 'RegularizedChangeGenerator':
        generator = RegularizedChangeGenerator(model, image, target, alpha=5, omega=10).to(device)
        optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)

    elif method == 'SmoothChangeGenerator':
        generator = SmoothChangeGenerator(model, image, target, kernel_size=9, sigma=2.0).to(device)
        optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)

    elif method == 'DeformationGenerator':
        generator = DeformationGenerator(model, image, target).to(device)
        optimizer = torch.optim.Adam([generator.parameter], lr=1e-1)

    elif method == 'AdversarialGenerator':
        generator = AdversarialGenerator(model, image, target).to(device)
        generator.load_adversarial("23_test_denoiser")
        optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)

    elif method == 'DetectionAdversarialGenerator':
        generator = DetectionAdversarialGenerator(model, image, target).to(device)
        generator.load_adversarial("23_test_denoiser")
        optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)

    else:
        raise ValueError(f"{method} not found!")

    generator.generate(optimizer, 100, verbose=True)

    new_image, _ = generator.adapt()
    new_prediction = model(new_image)[:, 1] > .5

    return new_image, prediction, new_prediction


def eval(name: str,
         slice: str,
         output_file: str,
         image: torch.Tensor,
         target: torch.Tensor,
         new_image: torch.Tensor,
         prediction: torch.Tensor,
         new_prediction: torch.Tensor):

    iou_before = intersection_over_union(prediction, target)
    iou_after = intersection_over_union(new_prediction, target)
    pred_size = torch.sum(prediction).item()
    new_pred_size = torch.sum(new_prediction).item()
    target_size = torch.sum(target).item()
    change = torch.sum(torch.abs(image - new_image)).item()

    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(output_file)

    # Write or append the row
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header if the file is new
        if not file_exists:
            writer.writerow([
                "name", "slice",
                "iou_before", "iou_after",
                "pred_size", "new_pred_size", "target_size",
                "change"
            ])

        # Write the data row
        writer.writerow([
            name,
            slice,
            iou_before,
            iou_after,
            pred_size,
            new_pred_size,
            target_size,
            change
        ])


def main(method: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0).to(device)
    dataset = Dataset2D("data/Dataset101_fcd")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    output_file = f"{method}_{time.strftime("%Y%m%d-%H%M%S")}_evaluation.csv"

    for patient, i, image, target in tqdm(dataloader, desc='evaluation'):
        image = image.to(device)
        target = target.to(device)
        new_image, prediction, new_prediction = generate(method, model, image, target, device)
        if new_image is None:
            continue
        eval(patient[0], i.item(), output_file, image[0], target[0], new_image, prediction, new_prediction)


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with call arguments
    """
    parser = argparse.ArgumentParser(description="evaluation")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="RegularizedChangeGenerator",
        help="Name of the method to evaluate",
    )

    return parser.parse_args()


if __name__ == '__main__':
    method = get_args().method

    main(method)
