from typing import Dict, List, Optional, Set, Tuple, Union
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTraining

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class MAE(ViTMAEForPreTraining):
    '''
    https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.TFViTMAEForPreTraining
    '''

    def __init__(
        self, 
        config: ViTMAEConfig, 
    ) -> None:
        
        super().__init__(config)
        self.config = config

        self.cpu_device = torch.device("cpu")
        
    def merge_patches(self, true_patches, pred_patches, is_masked):
        true_patches = true_patches * (1 - is_masked[:, :, None])
        pred_patches = pred_patches * is_masked[:, :, None]
        mixed_image = self.unpatchify(true_patches + pred_patches)
        return mixed_image

    def save_img(self, pixels, to):
        pixels = pixels.squeeze().detach()
        if pixels.device != self.cpu_device:
            pixels = pixels.cpu()
        pixels = (pixels + 1.) / 2.

        cm = plt.get_cmap('hot')
        pixels = cm(pixels)

        pixels = (pixels[:, :, :3] * 255.).astype(np.uint8)
        Image.fromarray(pixels).resize((1024, 1024)).save(to)

    def get_nth_boxed_visualisation(self, res, idx):
        square_num_patch = self.config.image_size // self.config.patch_size

        patch_box = torch.zeros((self.config.patch_size, self.config.patch_size))
        patch_box[:, -1] = patch_box[:, 0] = patch_box[0, :] = patch_box[-1, :] = 1
        patch_boxes = patch_box.repeat(square_num_patch, square_num_patch)

        mixed_image = res["mixed_image"][idx].squeeze()
        mask = res["mask"][idx].reshape(square_num_patch, square_num_patch)

        upscaled_mask = mask.repeat_interleave(self.config.patch_size, 0).repeat_interleave(self.config.patch_size, 1)
        colored_patch_boxes = (1 - upscaled_mask.cpu()) * patch_boxes

        return (mixed_image.detach().cpu() + colored_patch_boxes).clip(-1, 1)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> dict:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        reconstructed_patch_sequence = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        mixed_image = self.merge_patches(
            true_patches=self.patchify(pixel_values),
            pred_patches=reconstructed_patch_sequence,
            is_masked=mask
        )

        loss = self.forward_loss(pixel_values, reconstructed_patch_sequence, mask)
        # target = self.patchify(pixel_values)
        # if self.config.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.0e-6) ** 0.5

        # loss = (reconstructed_patch_sequence - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return {
            "loss": loss,
            "logits": reconstructed_patch_sequence,
            "mask": mask,
            "ids_restore": ids_restore,
            "hidden_states": outputs.hidden_states,
            "example_mixed_image": mixed_image[0],
            "mixed_image": mixed_image,
        }

        output = (logits, mask, ids_restore) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )