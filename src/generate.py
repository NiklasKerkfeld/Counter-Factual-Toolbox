import torch
from torch.nn import CrossEntropyLoss
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

    model = get_network(configuration='2d', fold=0)
    loss = MaskedCrossentropy()
    # generator = ElasticDeformation2D(model, (160, 256), (20, 32), loss=loss, alpha=.001)
    # optimizer = torch.optim.Adam([generator.dx, generator.dy], lr=1e-1)
    generator = ChangeGenerator(model, (2, 160, 256), loss=loss, alpha=10.0)
    optimizer = torch.optim.Adam([generator.change], lr=1e-1)

    main(generator, optimizer)
