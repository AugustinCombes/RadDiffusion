import os
pjoin = os.path.join

import logging
import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def save_img(pixels, to):
    pixels = pixels.cpu().numpy()
    if pixels.shape[-1] != 3:
        pixels = plt.get_cmap('hot')(pixels)
    pixels = (pixels * 255.).astype(np.uint8)
    Image.fromarray(pixels).resize((1024, 1024)).convert('RGB').save(to)

def save_pt_img(pixels, to):
    pixels = pixels.squeeze().detach()
    pixels = pixels.cpu()
    pixels = (pixels + 1.) / 2.
    save_img(pixels, to)

def get_nth_boxed_visualisation(config, image, mask=None, save=False):
    image_size = config.image_size
    patch_size = config.patch_size
    square_num_patch = image_size // patch_size

    patch_box = torch.zeros((patch_size, patch_size))
    patch_box[:, -1] = patch_box[:, 0] = patch_box[0, :] = patch_box[-1, :] = 1
    patch_boxes = patch_box.repeat(square_num_patch, square_num_patch)

    mixed_image = image.squeeze()
    mask = mask.reshape(square_num_patch, square_num_patch)
    upscaled_mask = mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    colored_patch_boxes = (1 - upscaled_mask.cpu()) * patch_boxes

    output = (mixed_image.detach().cpu() + colored_patch_boxes).clip(-1, 1)
    if not save:
        return output
    save_pt_img(output, save)

def initialize_logs(run_name):
    os.mkdir(run_name)
    for idx in range(5):
        os.mkdir(os.path.join(run_name, f"vis_{idx}"))
    logging.basicConfig(filename=f'{run_name}/logs.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_results(txt_path, names, auroc_list, auprc_list, f1_list):
    with open(txt_path, "w") as f:
        f.write("#*# Linear probing #*#")
        f.write('\n')
        for idx, name in enumerate(names):
            f.write(name)
            f.write(': AUROC {:.2f}%, AUPRC {:.2f}%, F1 {:.2f}%'.format(auroc_list[idx] * 100, auprc_list[idx] * 100, f1_list[idx] * 100))
            f.write('\n')
        f.write("mean")
        f.write(': AUROC {:.2f}%, AUPRC {:.2f}%, F1 {:.2f}%'.format(auroc_list[-1] * 100, auprc_list[-1] * 100, f1_list[-1] * 100))