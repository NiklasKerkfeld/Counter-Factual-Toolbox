import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange

from src.Architecture.CustomLoss import MaskedCrossentropy
from src.Architecture.Generator import Generator
from src.utils import get_network, load_data


def main(generator: Generator, optimizer: Adam, steps: int = 100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    print(f"Using device: {device}")

    item = load_data('data/Dataset101_fcd/sub-00001', device=device, slice=144)
    image = item['tensor'][None].to(device)
    target = item['target'].to(device)

    # grid = F.affine_grid(torch.tensor([[[1.0, -0.0, 0.1], [0.0, 1.0, 0.0]], [[1.0, -0.0, 0.1], [0.0, 1.0, 0.0]]], device=device), [2, 1, 160, 256])
    # image = F.grid_sample(image.permute(1, 0, 2, 3), grid, padding_mode='reflection').permute(1, 0, 2, 3)

    print("starting process...")
    bar = trange(steps, desc='generating...')
    for _ in bar:
        optimizer.zero_grad()
        loss = generator(image, target)
        loss.backward()
        optimizer.step()
        bar.set_description(f"loss: {loss.detach().cpu().item()}")

    loss = CrossEntropyLoss()
    with torch.no_grad():
        new_image, cost = generator.adapt(image)
        original_prediction = generator.model(image)
        deformed_prediction = generator.model(new_image)

        original_loss = loss(original_prediction, target)
        deformed_loss = loss(deformed_prediction, target)

    print(f"Reduced loss from {original_loss} to {deformed_loss} with an adaption that has a cost of: {cost}.")

    generator.visualize(image, target)


if __name__ == '__main__':
    from src.Architecture.ChangeGenerator import ChangeGenerator
    from src.Architecture.DeformationGenerator import ElasticDeformation2D
    from src.Architecture.AffineGenerator import AffineGenerator

    model = get_network(configuration='2d', fold=0)
    loss = MaskedCrossentropy()
    # generator = ElasticDeformation2D(model, (160, 256), (20, 32), loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.dx, generator.dy], lr=1e-1)
    generator = AffineGenerator(model, loss=loss, alpha=1.0)
    optimizer = torch.optim.Adam([generator.change], lr=1e-1)

    main(generator, optimizer)

    print(generator.theta)
