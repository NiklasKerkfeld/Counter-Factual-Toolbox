import torch
from matplotlib import pyplot as plt
from tqdm import trange

from src.Architecture.DeformationGenerator import ElasticDeformation2D
from src.Framework.utils import get_network, load_data


def main(steps: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    generator = ElasticDeformation2D(model, (256, 256), (16, 16))
    generator.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    item = load_data('data/Dataset101_fcd/sub-00001', device=device, slice=32)
    image = item['tensor'].to(device)
    target = item['target'].to(device)

    print("starting process...")

    bar = trange(steps, desc='generating...')
    for _ in bar:
        optimizer.zero_grad()
        loss = generator(image, target)
        loss.backward()
        optimizer.step()
        bar.set_description(f"loss: {loss.detach().cpu().item()}")

    new_image = generator.image(image)

    # Plotting
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Deformed")
    plt.imshow(new_image, cmap='gray')
    plt.savefig("logs/result.png")

    
   
if __name__ == '__main__':
    main(steps=100)
