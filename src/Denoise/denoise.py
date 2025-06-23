import torch
from monai.networks.nets import BasicUNet
from tqdm import trange

from src.utils import get_image


def denoise(image: torch.Tensor, alpha: float =.01) -> torch.Tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128))
    model.load_state_dict(torch.load("models/34_test_adversarial.pth"))
    model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        for _ in trange(100):
            pred = model(image)
            image -= alpha * pred

    return image.cpu()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image, _ = get_image('data/Dataset101_fcd/sub-00003', 2)

    image = image
    noise = torch.randn_like(image)

    reconstruction = denoise(image + noise)

    print(f"noise applied: {torch.abs(noise).sum()}")
    print(f"noise left: {torch.abs(image - reconstruction).sum()}")

    plt.figure(figsize=(10, 15))
    plt.subplot(1, 3, 1)
    plt.title("Original image")
    plt.imshow(image[0, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Noised image")
    plt.imshow((image + noise)[0, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed image")
    plt.imshow(reconstruction[0, 0], cmap='gray')
    plt.axis('off')

    plt.show()