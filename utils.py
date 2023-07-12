import os

import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

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
