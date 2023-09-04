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
            # hidden_dropout_prob=0.3,
        )

        self.cpu_device = torch.device("cpu")

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size, self.disc_config.hidden_size),
            ViTMAEWrapper(ViTMAELayer(self.disc_config)),
            # ViTMAEWrapper(ViTMAELayer(self.disc_config)),
            torch.nn.LayerNorm(self.disc_config.hidden_size, self.disc_config.layer_norm_eps),
            torch.nn.Linear(self.disc_config.hidden_size, 1),
            # torch.nn.Sigmoid()
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

    def sampler(self, sequence, len_keep=None, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise. Allows the use of specific sampling sequence len_keep.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
            len_keep 
        """
        batch_size, seq_length, dim = sequence.shape
        if len_keep is None:
            len_keep = int(seq_length * (1 - self.config.mask_ratio))
        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

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

        The goal of the discriminator is to retrieve mask, that is to predict high probabilities for
            masked positions and low probabilities for unmasked ones.
        '''

        full_embedding_sequence = self.vit.embeddings.patch_embeddings(mixed_image)
        full_embedding_sequence += self.vit.embeddings.position_embeddings[:, 1:, :]

        batch_size, seq_length, _ = full_embedding_sequence.shape
        L_true = int(seq_length * (1 - self.config.mask_ratio))
        L_fake = int(seq_length * self.config.mask_ratio)
        L_target = seq_length // 4

        sequence_true = full_embedding_sequence[~mask.bool()].reshape(
            (batch_size, L_true, self.config.hidden_size)
            )
        # sequence_unmasked_true, _, _ = self.vit.embeddings.random_masking(sequence_true)
        sequence_unmasked_true, _, _ = self.sampler(sequence_true, L_target // 2)

        sequence_fake = full_embedding_sequence[mask.bool()].reshape(
            (batch_size, L_fake, self.config.hidden_size)
            )
        # sequence_unmasked_fake, _, _ = self.vit.embeddings.random_masking(sequence_fake)
        sequence_unmasked_fake, _, _ = self.sampler(sequence_fake, L_target // 2)

        sequence_unmasked = torch.concat((sequence_unmasked_true, sequence_unmasked_fake), dim=1)
        masked_embeddings = self.vit.encoder(sequence_unmasked)['last_hidden_state']
        masked_embeddings = self.vit.layernorm(masked_embeddings)

        y_hat = self.discriminator(masked_embeddings)
        y = torch.concat((torch.ones_like(sequence_unmasked_true[..., :1]), torch.zeros_like(sequence_unmasked_fake[..., :1])), dim=1)        
        return y, y_hat

    def compute_gamma(self, l2_loss, adv_loss):
        '''
        Computes \gamma = \frac{grad_last(l_2)}{grad_last(l_adv) + 1e-6}.

        The function grad_last is the norm of the gradients of the final linear projection 
            to the image space, w.r.t the loss taken as argument.
        '''

        last_layer = self.decoder.decoder_pred
        last_layer_weight = last_layer.weight

        l2_grads = torch.norm(
            torch.autograd.grad(l2_loss, last_layer_weight, retain_graph=True)[0]
            )
        adv_grads = torch.norm(
            torch.autograd.grad(adv_loss, last_layer_weight, retain_graph=True)[0]
            )
        
        gamma = torch.norm(l2_grads) / (adv_grads + 1e-4)
        gamma = torch.clamp(gamma, 0, 1e4).detach()
        return gamma, l2_grads.detach(), adv_grads.detach() #gamma * 0.8

        # self.vit.zero_grad()
        # l2_loss.backward(retain_graph=True)
        # grads_l2 = [param.grad for param in self.vit.parameters()]
        # l2_norm = sum([torch.norm(g) for g in grads_l2 if g is not None])

        # self.vit.zero_grad()
        # adv_loss.backward(retain_graph=True)
        # grads_adv = [param.grad for param in self.vit.parameters()]
        # adv_norm = sum([torch.norm(g) for g in grads_adv if g is not None])

        # self.vit.zero_grad()
        # gamma = l2_norm / (adv_norm + 1e-6)

        # return gamma
    
        # self.vit.zero_grad()
        # l2_loss.backward(retain_graph=True)
        # grads_l2 = [param.grad for param in self.vit.parameters()]
        # l2_norm = sum([torch.norm(g) for g in grads_l2 if g is not None])

        # self.vit.zero_grad()
        # adv_loss.backward(retain_graph=True)
        # grads_adv = [param.grad for param in self.vit.parameters()]
        # adv_norm = sum([torch.norm(g) for g in grads_adv if g is not None])

        # self.vit.zero_grad()
        # alpha = l2_norm / (l2_norm + adv_norm)

        # return alpha

        # l2_loss.backward(retain_graph=True)
        # grad_l2 = model.decoder.decoder_pred.weight.grad
        # model.zero_grad()

        # adv_loss.backward(retain_graph=True)
        # grad_adv = model.decoder.decoder_pred.weight.grad
        # model.zero_grad()

        # gamma = grad_l2.norm() / (grad_adv.norm() + 1e-6)
        # return gamma

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