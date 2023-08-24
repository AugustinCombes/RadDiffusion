from typing import Dict, List, Optional, Set, Tuple, Union
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTraining
from transformers import ViTMAELayer, ViTMAEConfig

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ViTMAEWrapper(torch.nn.Module):
    def __init__(self, layer):
        super(ViTMAEWrapper, self).__init__()
        self.layer = layer

    def forward(self, x):
        output_tuple = self.layer(x)
        return output_tuple[0]

class MAE_GAN(ViTMAEForPreTraining):
    '''
    https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.TFViTMAEForPreTraining
    '''

    def __init__(
        self, 
        config: ViTMAEConfig, 
    ) -> None:
        
        super().__init__(config)
        self.config = config
        self.disc_config = ViTMAEConfig(
            hidden_size = config.decoder_hidden_size,
            num_hidden_layers = config.decoder_num_hidden_layers,
            num_attention_heads = config.decoder_num_attention_heads,
            intermediate_size = config.decoder_intermediate_size,
        )

        self.cpu_device = torch.device("cpu")

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size, self.disc_config.hidden_size),
            ViTMAEWrapper(ViTMAELayer(self.disc_config)),
            torch.nn.LayerNorm(self.disc_config.hidden_size, self.disc_config.layer_norm_eps),
            torch.nn.Linear(self.disc_config.hidden_size, 1),
            torch.nn.Sigmoid()
        )
        
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

    def forward_discriminator_loss(self, mixed_image, mask):
        '''
        Computes y_hat = D(\hat X).
        Let the mixed_image have L patches, mask_ratio % of them are generated patches and
            (1 - mask_ratio) % are original patches.
        
        We randomly mask a proportion of mask_ratio patches from the generated patches.
        Independently, we randomly mask a proportion of mask_ratio % patches from the original patches.

        Before concatenation, there is [mask_ratio * (1 - mask_ratio) * L] generated patches and
            [(1 - mask_ratio) * (1 - mask_ratio) * L] original patches.
        Thus, after concatenation, we get 
            [mask_ratio * (1 - mask_ratio) * L + (1 - mask_ratio) * (1 - mask_ratio) * L]
            = [(1 - mask_ratio) * L] patches.
        
        These patches are forwarded in the ViT encoder. The resulting representations are forwarded in
            the ViT discriminator, resulting in the logits y_hat.
        '''

        full_embedding_sequence = self.vit.embeddings.patch_embeddings(mixed_image)
        full_embedding_sequence += self.vit.embeddings.position_embeddings[:, 1:, :]

        batch_size, seq_length, _ = full_embedding_sequence.shape
        L_true = int(seq_length * (1 - self.config.mask_ratio))
        L_fake = int(seq_length * self.config.mask_ratio)

        sequence_true = full_embedding_sequence[~mask.bool()].reshape(
            (batch_size, L_true, self.config.hidden_size)
            )
        sequence_unmasked_true, _, _ = self.vit.embeddings.random_masking(sequence_true)

        sequence_fake = full_embedding_sequence[mask.bool()].reshape(
            (batch_size, L_fake, self.config.hidden_size)
            )
        sequence_unmasked_fake, _, _ = self.vit.embeddings.random_masking(sequence_fake)

        sequence_unmasked = torch.concat((sequence_unmasked_true, sequence_unmasked_fake), dim=1)
        masked_embeddings = self.vit.encoder(sequence_unmasked)['last_hidden_state']
        masked_embeddings = self.vit.layernorm(masked_embeddings)

        y_hat = self.discriminator(masked_embeddings)
        y = torch.concat((torch.ones_like(sequence_unmasked_true[..., :1]), torch.zeros_like(sequence_unmasked_fake[..., :1])), dim=1)        
        return y, y_hat

    def compute_gamma(model, l2_loss, adv_loss):
        '''
        Computes \gamma = \frac{grad_last(l_2)}{grad_last(l_adv) + 1e-6}.

        The function grad_last is the norm of the gradients of the final linear projection 
            to the image space, w.r.t the loss taken as argument.
        '''

        l2_loss.backward(retain_graph=True)
        grad_l2 = model.decoder.decoder_pred.weight.grad
        model.zero_grad()

        adv_loss.backward(retain_graph=True)
        grad_adv = model.decoder.decoder_pred.weight.grad
        model.zero_grad()

        gamma = grad_l2.norm() / (grad_adv.norm() + 1e-6)
        return gamma

    def forward(
        self,
        batch,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> dict:

        pixel_values = batch['image'][:, None, :, :]

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

        return {
            "loss": loss,
            "logits": reconstructed_patch_sequence,
            "mask": mask,
            "ids_restore": ids_restore,
            "hidden_states": outputs.hidden_states,
            "example_mixed_image": mixed_image[0],
            "mixed_image": mixed_image,
        }