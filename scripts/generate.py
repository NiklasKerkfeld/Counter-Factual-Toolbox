import torch

from src import generate
from src.utils import get_network
from src.Architecture.Generator import (ChangeGenerator,
                                        ElasticDeformation2D,
                                        AffineGenerator,
                                        AdversarialGenerator,
                                        DifferenceAdversarialGenerator,
                                        ScaleAndShiftGenerator,
                                        AffineGenerator)

if __name__ == '__main__':
    model = get_network(configuration='2d', fold=0)

    # generator = ScaleAndShiftGenerator(model, (1, 2, 160, 256), loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.scale, generator.shift], lr=1e-3)

    # generator = AffineGenerator(model, loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.change], lr=1e-3)

    # generator = ElasticDeformation2D(model, (1, 2, 160, 256), (20, 32), alpha=0.1)
    # optimizer = torch.optim.Adam([generator.dx, generator.dy], lr=1e-1)

    generator = ChangeGenerator(model, (1, 2, 160, 256), alpha=10.0)
    optimizer = torch.optim.Adam([generator.change], lr=1e-3)

    # generator = AdversarialGenerator(model, (1, 2, 160, 256))
    # generator.load_adversarial("adversarial")
    # optimizer = torch.optim.Adam([generator.change], lr=1e-2)

    # change_list = [torch.load("Results/69_AdversarialGenerator/bias_map.pt")]
    # generator = DifferenceAdversarialGenerator(model, (1, 2, 160, 256), change_list=change_list, alpha=10.0)
    # generator.load_adversarial()
    # optimizer = torch.optim.Adam([generator.change], lr=1e-3)

    # interesting: sub-00003, sub-00043, sub-00048, sub-00116, sub-00137
    generate('data/Dataset101_fcd/sub-00003', generator, optimizer)
