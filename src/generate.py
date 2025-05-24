import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from tqdm import trange

from src.Architecture.DeformationGenerator import ElasticDeformation2D
from src.Framework.utils import get_network, load_data
from src.utils import visualize_deformation_field


def main(steps: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_network(configuration='2d', fold=0)
    generator = ElasticDeformation2D(model, (160, 256), (20, 32), alpha=.001)
    generator.to(device)

    optimizer = torch.optim.Adam([generator.dx, generator.dy], lr=1e-1)

    item = load_data('data/Dataset101_fcd/sub-00001', device=device, slice=144)
    image = item['tensor'][None].to(device)
    target = item['target'].to(device)
    print(f"{target.max()=}")

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
        original_prediction = model(image)
        deformed_prediction = model(new_image)

        original_loss = loss(original_prediction, target)
        deformed_loss = loss(deformed_prediction, target)

        original_prediction = F.softmax(original_prediction, dim=1)[0, 1].cpu()
        deformed_prediction = F.softmax(deformed_prediction, dim=1)[0, 1].cpu()

    print(f"Reduced loss from {original_loss} to {deformed_loss} with an adaption that has a cost of: {cost}.")

    image = image[0, 0].cpu()
    new_image = new_image[0, 0].cpu()
    target = target.cpu()

    visualize_deformation_field(image,
                                generator.dx[0, 0].detach().cpu(),
                                generator.dy[0, 0].detach().cpu(),
                                scale=1)

    # Plotting
    plt.subplot(2, 3, 2)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Deformed")
    plt.imshow(new_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Target")
    plt.imshow(target[0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Original prediction")
    plt.imshow(original_prediction, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Deformed prediction")
    plt.imshow(deformed_prediction, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("logs/result.png", dpi=750)


if __name__ == '__main__':
    main(steps=100)
