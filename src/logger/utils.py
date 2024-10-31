import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(Image.open(buf))

    plt.close()

    return image


def process_images(
    images: torch.Tensor, dest_path: str, prefix: str = "", img_format: str = "jpeg"
):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param prefix: is the prefix to add to file names
    :param img_format: is the image format
    """

    # Create the destination folder
    os.makedirs(dest_path, exist_ok=True)

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).detach().numpy()
    # images = (images * 255).astype(np.uint8)
    # # Save images
    # for i, img in enumerate(images):
    #     save_path = os.path.join(dest_path, f"{prefix}{i:05}.{img_format}")
    #     plt.imsave(save_path, img)
    return images
