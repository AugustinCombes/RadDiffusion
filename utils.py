import os

import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

from PIL import Image

def plot_original(pixel_values, mask, config):
    patch_width = config.image_size // config.patch_size
    pixel_values = pixel_values.squeeze().cpu()
    pixel_values = (1 + pixel_values)/2
    mask = np.repeat(np.repeat(mask.cpu().numpy().reshape((patch_width, patch_width)), config.patch_size, axis=1), config.patch_size, axis=0)
    mask = 1 - mask
    pixel_values = (pixel_values * mask).clip(0, 1).squeeze()
    return pixel_values

def plot_reconstruction(reconstruction, mask, config):
    patch_width = config.image_size // config.patch_size
    reconstruction = reconstruction.detach().squeeze().cpu()
    reconstruction = (1 + reconstruction)/2
    mask = np.repeat(np.repeat(mask.cpu().numpy().reshape((patch_width, patch_width)), config.patch_size, axis=1), config.patch_size, axis=0)
    reconstruction = (reconstruction * mask).clip(0, 1).squeeze()
    return reconstruction

def plot_images_from_batch(image, mask, reconstruction, rows, config, display_test=False, save_to=None):
    fig, axs = plt.subplots(rows, 3, figsize=(10, 5*rows))
    axs = axs.reshape(rows, -1) if rows == 1 else axs
    
    for i in range(rows):
        origin = plot_original(image[i], mask[i] if not display_test else torch.zeros_like(mask[i]), config)
        axs[i, 0].imshow(origin, cmap='hot', vmin=0, vmax=1)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f"Original {i+1}")
        
        # Plot the reconstructed image
        recons = plot_reconstruction(reconstruction[i], mask[i], config)
        axs[i, 1].imshow(recons, cmap='hot', vmin=0, vmax=1)
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f"Reconstructed {i+1}")
        
        if display_test:
            origin = plot_original(image[i], mask[i], config)
        total = recons + origin
        axs[i, 2].imshow(total, cmap='hot', vmin=0, vmax=1)
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f"Total {i+1}")
        
    plt.tight_layout()
    
    if save_to is not None and isinstance(save_to, str):
        directory = os.path.dirname(save_to)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        fig.canvas.draw()
        plt.savefig(save_to)
    else:
        plt.show()
    
def plot_global_comparison(input_image, reconstructed_image, save_to=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    input_image, reconstructed_image = input_image.detach().cpu().squeeze(), reconstructed_image.detach().cpu().squeeze()

    axs[0].imshow(input_image, cmap='hot', vmin=0, vmax=1)
    axs[1].imshow(reconstructed_image, cmap='hot', vmin=0, vmax=1)
    axs[0].axis('off')
    axs[1].axis('off')

    if save_to is not None and isinstance(save_to, str):
        directory = os.path.dirname(save_to)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(save_to)
    else:
        plt.show()

def save_pt_img(pixels, to):
    pixels = pixels.squeeze().detach()
    pixels = pixels.cpu()
    pixels = (pixels + 1.) / 2.

    cm = plt.get_cmap('hot')
    pixels = cm(pixels)

    pixels = (pixels[:, :, :3] * 255.).astype(np.uint8)
    Image.fromarray(pixels).resize((1024, 1024)).save(to)

def get_nth_boxed_visualisation(image_size, patch_size, res, idx):
    square_num_patch = image_size // patch_size

    patch_box = torch.zeros((patch_size, patch_size))
    patch_box[:, -1] = patch_box[:, 0] = patch_box[0, :] = patch_box[-1, :] = 1
    patch_boxes = patch_box.repeat(square_num_patch, square_num_patch)

    mixed_image = res["mixed_image"][idx].squeeze()
    mask = res["mask"][idx].reshape(square_num_patch, square_num_patch)

    upscaled_mask = mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    colored_patch_boxes = (1 - upscaled_mask.cpu()) * patch_boxes

    return (mixed_image.detach().cpu() + colored_patch_boxes).clip(-1, 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)