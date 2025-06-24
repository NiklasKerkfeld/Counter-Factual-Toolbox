import glob

import torch

from src.utils import get_network, get_image
from src.Architecture.Generator import (ChangeGenerator,
                                        ElasticDeformation,
                                        AffineGenerator,
                                        AdversarialGenerator,
                                        DifferenceAdversarialGenerator,
                                        ScaleAndShiftGenerator,
                                        AffineGenerator)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('data/Dataset101_fcd/sub-00003', 2)

    # generator = ScaleAndShiftGenerator(model, (1, 2, 160, 256), loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.scale, generator.shift], lr=1e-3)

    # generator = AffineGenerator(model, loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.change], lr=1e-3)

    generator = ElasticDeformation(model, image, target)
    optimizer = torch.optim.Adam([generator.dx, generator.dy], lr=1e-1)

    # generator = ChangeGenerator(model, image, target, alpha=1.0, kernel_size=9, sigma=2)
    # optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)

    # generator = AdversarialGenerator(model, image, target, alpha=1.0)
    # generator.load_adversarial("23_test_adversarial")
    # optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)

    # change_list = [torch.load("Results/69_AdversarialGenerator/bias_map.pt")]
    # generator = DifferenceAdversarialGenerator(model, (1, 2, 160, 256), change_list=change_list, alpha=10.0)
    # generator.load_adversarial()
    # optimizer = torch.optim.Adam([generator.change], lr=1e-3)

    # interesting: sub-00003, sub-00043, sub-00048, sub-00116, sub-00137

    generator.name = f"{len(glob.glob('Results/*'))}_{generator.__class__.__name__}"
    generator.to(device)

    generator.generate(optimizer, 100)

    print("\nstarting logging...\n")
    generator.log_and_visualize('GradCAM')