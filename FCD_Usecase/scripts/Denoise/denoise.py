import torch
from monai.networks.nets import BasicUNet
from tqdm import trange

from FCD_Usecase.scripts.utils.utils import get_image


def denoise(image: torch.Tensor, noise, alpha: float =.01) -> torch.Tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BasicUNet(in_channels=2,
                      out_channels=2,
                      spatial_dims=2,
                      features=(64, 128, 256, 512, 1024, 128))
    model.load_state_dict(torch.load("models/23_test_denoiser.pth", map_location=device))
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        init_pred = model(image)

    plt.figure(figsize=(5, 5))
    plt.title("Original image prediction")
    plt.imshow(init_pred[0, 0], cmap='RdBu')
    plt.axis('off')
    plt.savefig('init_pred.png', dpi=750)
    plt.close()

    image += noise

    model.eval()
    change = []
    with torch.no_grad():
        bar = trange(500)
        for _ in bar:
            pred = model(image) - init_pred
            image -= alpha * pred

            change.append(torch.abs(pred).sum())
            bar.set_description(f"change: {change[-1]}")

    plt.plot(change)
    plt.tight_layout()
    plt.savefig('change_curve.png', dpi=750)

    return image.cpu()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image, _ = get_image('data/Dataset101_fcd/sub-00003', 2)

    noise = torch.randn_like(image) * 0.1

    reconstruction = denoise(image.clone(), noise)

    print(f"noise applied: {torch.abs(noise).sum()}")
    print(f"noise left: {torch.abs(image - reconstruction).sum()}")

    plt.figure(figsize=(15, 5))
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

    plt.tight_layout()
    plt.savefig(f"denoise_result.png", dpi=750)
    plt.close()