from typing import Dict, List, Optional, Set, Tuple, Union
import math

import torch
from torch import nn
from einops import rearrange

from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTForMaskedImageModeling

from models.ConvT_generator import GlobalDecoder

class InViT(ViTForMaskedImageModeling):
    '''
    https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.TFViTMAEForPreTraining
    '''

    def __init__(
        self, 
        config: ViTConfig, 
        do_global_reconstruction = False,
        do_patchwise_deconvolution = False
    ) -> None:
        
        super().__init__(config)

        self.do_patchwise_deconvolution = do_patchwise_deconvolution
        if not do_patchwise_deconvolution:
            # transformers original code
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=config.hidden_size,
                    out_channels=config.encoder_stride**2 * config.num_channels,
                    kernel_size=1,
                ),
                nn.PixelShuffle(config.encoder_stride),
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 32, (4, 4), stride=4),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.ConvTranspose2d(32, 8, (4, 4), stride=4),
                nn.BatchNorm2d(8),
                nn.GELU(),
                nn.ConvTranspose2d(8, 1, (2, 2), stride=2),
                nn.Tanh(),
            )

        self.do_global_reconstruction = do_global_reconstruction
        if do_global_reconstruction:
            self.global_decoder = GlobalDecoder(config)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
    ) -> tuple:
        
        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        cls_output = sequence_output[:, 1]
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        size = self.config.image_size // self.config.patch_size
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct patch pixel values
        if self.do_patchwise_deconvolution:
            sequence_output = rearrange(sequence_output, "b c h w -> (b h w) c 1 1")
            sequence_output = self.decoder(sequence_output)
            reconstructed_pixel_values = rearrange(sequence_output, "(b h w) u p1 p2 -> b u (p1 h) (p2 w)", 
                                        u=self.config.num_channels, p1=self.config.patch_size, p2=self.config.patch_size,
                                        h=size, w=size
                                        )
        else:
            reconstructed_pixel_values = self.decoder(sequence_output)

        masked_patch_loss = None
        if bool_masked_pos is not None:
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            patch_reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_patch_loss = (patch_reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        result = {
            "patchwise_loss": masked_patch_loss,
            "patchwise_reconstructions": reconstructed_pixel_values,
            "hidden_states": outputs.hidden_states,
        }

        if self.do_global_reconstruction:
            reconstructed_image = self.global_decoder(cls_output[..., None, None])
            global_reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_image)

            result["cls_reconstructed_image"] = reconstructed_image
            result["global_reconstruction_loss"] = global_reconstruction_loss

        return result