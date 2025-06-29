import torch

from CounterFactualToolbox.Generator import *
from FCD_Usecase.scripts.utils.utils import get_network, get_image


def test_smoothchangegenerator():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('FCD_Usecase/data/Dataset101_fcd/sub-00003', 2)

    generator = SmoothChangeGenerator(model, image, target)
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)
    generator.to(device)

    generator.generate(optimizer, 10)


def test_regularizedchangegenerator():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('FCD_Usecase/data/Dataset101_fcd/sub-00003', 2)

    generator = RegularizedChangeGenerator(model, image, target)
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)
    generator.to(device)

    generator.generate(optimizer, 10)


def test_deformationgenerator():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('FCD_Usecase/data/Dataset101_fcd/sub-00003', 2)

    generator = DeformationGenerator(model, image, target)
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)
    generator.to(device)

    generator.generate(optimizer, 10)


def test_adversarialgenerator():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('FCD_Usecase/data/Dataset101_fcd/sub-00003', 2)

    generator = AdversarialGenerator(model, image, target, alpha=1.0)
    generator.load_adversarial("23_test_denoiser")
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-3)
    generator.to(device)

    generator.generate(optimizer, 10)


def test_detectionadversarialgenerator():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('FCD_Usecase/data/Dataset101_fcd/sub-00003', 2)

    generator = DetectionAdversarialGenerator(model, image, target, alpha=1.0)
    generator.load_adversarial("23_test_denoiser")
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-3)
    generator.to(device)

    generator.generate(optimizer, 10)


def test_mulitple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    image, target = get_image('FCD_Usecase/data/Dataset101_fcd/sub-00003', 2)

    generator = SmoothChangeGenerator(model, image, target, alpha=1.0, kernel_size=9, sigma=2)
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)
    generator.to(device)
    generator.generate(optimizer, 10)

    previous = [generator.parameter.clone()]

    generator = SmoothChangeGenerator(model, image, target, alpha=2.0, kernel_size=9, sigma=2, previous=previous)
    optimizer = torch.optim.Adam([generator.parameter], lr=1e-2)
    generator.to(device)

    generator.generate(optimizer, 10)

