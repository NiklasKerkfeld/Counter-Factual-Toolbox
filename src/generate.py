import torch
from matplotlib import pyplot as plt
from tqdm import trange

from src.Architecture.DeformationGenerator import ElasticDeformation2D
from src.Framework.utils import get_network, load_data


def main(steps: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_network(configuration='2d', fold=0)
    generator = ElasticDeformation2D(model, (256, 256), (16, 16))
    generator.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    image = load_data('path/to/image', device=device, slice=32)

    for i in trange(steps):
        optimizer.zero_grad()
        loss = generator(image)
        loss.backward()
        optimizer.step()

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
    main()
