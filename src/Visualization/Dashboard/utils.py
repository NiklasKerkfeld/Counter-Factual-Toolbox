from PIL import Image
import numpy as np
import matplotlib.cm as cm


def blend_overlay(base: Image, target_mask_array: np.ndarray, cmap='Greens', alpha=0.5):
    # Normalize target mask and apply colormap (Greens)
    target_mask_norm = (target_mask_array > .05).astype(np.uint8)
    green_colormap = cm.get_cmap(cmap)

    # Apply colormap: returns RGBA float32 (0–1)
    colored_mask = (green_colormap(target_mask_array)[:, :, :3] * 255).astype(np.uint8)
    colored = Image.fromarray(colored_mask).convert("RGBA")

    # Create alpha mask where target is non-zero
    alpha_mask = (target_mask_norm * int(alpha * 255)).astype(np.uint8)
    alpha_mask_img = Image.fromarray(alpha_mask)
    colored.putalpha(alpha_mask_img)

    # Composite colored mask over base image
    blended = Image.alpha_composite(base.convert("RGBA"), colored)
    return blended


def pad(img: Image) -> Image:
    x, y = img.size
    size = max(x, y)
    new_img = Image.new('RGBA', (size, size), (0, 0, 0, 255))
    new_img.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_img

