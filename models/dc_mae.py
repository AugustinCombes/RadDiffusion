import torch
import torch.nn as nn
from einops import rearrange
import collections.abc
from copy import deepcopy
import math

from typing import Optional, Set, Tuple, Union
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings, ViTMAEPatchEmbeddings, get_2d_sincos_pos_embed
from transformers import PreTrainedModel, ViTMAEPreTrainedModel


class DeconViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1], image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=3, padding=1)

    def forward(self, pixel_values):
        if len(pixel_values.size()) == 3:
            pixel_values = pixel_values[:, None, :, :]
        
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        p, q = self.num_patches
        x = rearrange(pixel_values, "b c (p h) (q w) -> (b p q) c h w", p=p, q=q)
        x = self.proj(x)
        return rearrange(x, "(b p q) c h w -> b (p q) c h w", p=p, q=q)
    
class DeconViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = DeconViTMAEPatchEmbeddings(config)
        p, q = self.patch_embeddings.num_patches
        self.num_patches = p * q

        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.config.hidden_size, int(self.num_patches**0.5), add_cls_token=True
        )
        pos_embed = pos_embed[None, ..., None, None]

        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, channels, h, w = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep[..., None, None, None].repeat(1, 1, channels, h, w))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None):
        if len(pixel_values.size()) == 3:
            pixel_values = pixel_values[:, None, :, :]
            
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)


        # append cls token
        cls_token = self.cls_token[..., None, None] + self.position_embeddings[:, :1, :]
        _, _, _, ph, pw = embeddings.shape
        cls_tokens = cls_token.expand(batch_size, -1, -1, ph, pw)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
    
class DeconvolutiveAttention(nn.Module):
    def __init__(self, dim, num_heads, spatial_ratio, kernel_size=1):
        super(DeconvolutiveAttention, self).__init__()

        self.scale = dim ** -0.5
        self.num_heads = num_heads

        if spatial_ratio >= 1:
            self.rescale = nn.Upsample(scale_factor=spatial_ratio)
        else:
            self.rescale = nn.AvgPool2d(kernel_size=int(1/spatial_ratio))

        #omitted qkv bias in 2nd Conv2d
        self.nK = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, bias=False, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1)
            )
        self.nQ = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, bias=False, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1)
            )
        self.nV = nn.Sequential(
            self.rescale,
            nn.Conv2d(dim, dim, kernel_size=kernel_size, bias=False, groups=dim, padding=(kernel_size-1)//2),
            nn.Conv2d(dim, dim, 1)
        )
        
    def patchify(self, p):
        return lambda x: rearrange(x, "b p c h w -> (b p) c h w")
    
    def unpatchify(self, p):
        return lambda x: rearrange(x, "(b p) c h w -> b p c h w", p=p) if len(x.shape) == 4 else rearrange(x, "(b p) c -> b p c", p=p)
    
    def rearrange_for_multi_head_attention(self, hidden_state, p):
        return rearrange(hidden_state, "(b t) (m d) h w -> b m t d h w", m=self.num_heads, t=p)
        
    def forward(self, hidden_state):
        if len(hidden_state.size()) < 5:
            hidden_state = hidden_state[..., None, None]
        p = hidden_state.shape[1]

        hidden_state = self.patchify(p)(hidden_state)
        key = self.rearrange_for_multi_head_attention(self.nK(hidden_state), p)
        query = self.rearrange_for_multi_head_attention(self.nQ(hidden_state), p)
        key, query = key.mean(dim=(-2, -1)), query.mean(dim=(-2, -1))
        
        attention_score = torch.einsum("bmlk,bmtk->bmlt", [query, key]) * self.scale
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)

        value = self.rearrange_for_multi_head_attention(self.nV(hidden_state), p)
        context = torch.einsum("bmlt,bmtvhw->bmlvhw", [attention_probs, value])
        context = rearrange(context, "b m t d h w -> b t (m d) h w")

        hidden_state = self.rescale(hidden_state)
        context = self.unpatchify(p)(hidden_state) + context #skip connexion
        return context
    
class ConvProjection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, activation=None):
        super(ConvProjection, self).__init__()
        self.cff = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.activation = False if activation != "gelu" else nn.GELU()
        
    def forward(self, hidden_state):
        p = hidden_state.shape[1]
        hidden_state = rearrange(hidden_state, "b p c h w -> (b p) c h w")
        hidden_state = self.cff(hidden_state)
        if self.activation:
            hidden_state = self.activation(hidden_state)
        return rearrange(hidden_state, "(b p) c h w -> b p c h w", p=p)
    
class LN(nn.Module):
    def __init__(self, tuple_dim):
        super(LN, self).__init__()
        self.ln = nn.LayerNorm(tuple_dim)
        
    def forward(self, x):
        p = x.shape[1]
        if not len(x.shape) == 3:
            x = rearrange(x, "b p c h w -> (b p) c h w")
        x = self.ln(x)
        if not len(x.shape) == 3:
            x = rearrange(x, "(b p) c h w -> b p c h w", p=p)
        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hw, ratio, kernel_size=1):
        super(ConvDecoder, self).__init__()
        
        self.ln_before = LN((input_dim, hw, hw))# if hw!=1 else LN((input_dim))
        self.mhca = DeconvolutiveAttention(dim=input_dim, num_heads=num_heads, spatial_ratio=ratio, kernel_size=kernel_size)

        r = int(hw * ratio)
        self.ln_after = LN((input_dim, r, r))
        
        self.proj = ConvProjection(input_dim, input_dim, activation="gelu", kernel_size=kernel_size)
        self.proj_channel = ConvProjection(input_dim, output_dim, kernel_size=kernel_size)
        
    def forward(self, hidden_state):
        hidden_state = self.mhca(self.ln_before(hidden_state)) #contains skip connexion        
        hidden_state = hidden_state + self.proj(self.ln_after(hidden_state)) #2nd skip connexion
        return self.proj_channel(hidden_state)
    
class DeconViTMAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
                ConvDecoder(config.hidden_size, 
                            config.hidden_size, 
                            config.num_attention_heads, 
                            hw=2**(5-idx), 
                            ratio=0.5, 
                            kernel_size=2*(5-idx-1) + 1)
                for idx in range(5)
            ])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs

        return hidden_states
    
class DeconViTMAEModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = DeconViTMAEEmbeddings(config)
        self.encoder = DeconViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
    ) -> dict:
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)

        sequence_output = self.encoder(
            embedding_output,
        )
        sequence_output = sequence_output.squeeze()
        sequence_output = self.layernorm(sequence_output)

        return dict(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
        )
    
class DeconViTMAEDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = ConvProjection(config.hidden_size, config.decoder_hidden_size, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.num_patches = num_patches
        
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList([
            ConvDecoder(256, 128, 8, 1, 2, kernel_size=1),
            ConvDecoder(128, 64, 8, 2, 2, kernel_size=1),
            ConvDecoder(64, 32, 4, 4, 2, kernel_size=3),
            ConvDecoder(32, 16, 4, 8, 2, kernel_size=5),
            ConvDecoder(16, 1, 4, 16, 2, kernel_size=7),
        ])

        self.decoder_norm = LN((config.num_channels, config.patch_size, config.patch_size))
        self.decoder_pred = ConvProjection(
            config.num_channels, config.num_channels, kernel_size=1
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.config.decoder_hidden_size, int(num_patches**0.5), add_cls_token=True
        )
        decoder_pos_embed = decoder_pos_embed[None, ..., None, None]

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.from_numpy(decoder_pos_embed).float(), requires_grad=False
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        ids_restore,
    ):
        if len(hidden_states.size()) < 5:
            hidden_states = hidden_states[..., None, None]
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)[..., None, None]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore[..., None, None, None].repeat(1, 1, x.shape[2], 1, 1))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        for i, layer_module in enumerate(self.decoder_layers):
            hidden_states = layer_module(hidden_states)

        logits = self.decoder_norm(hidden_states)
        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        return dict(
            logits=logits,
            last_hidden_states=hidden_states,
        )

class ViTMAEForPreTraining(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = DeconViTMAEModel(config)
        self.decoder = DeconViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.sq_num_patches = int(math.sqrt(self.decoder.num_patches))

        # Initialize weights and apply final processing
        self.post_init()

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        x = rearrange(pixel_values, "b c (p h) (q w) -> b (p q) c h w", p=self.sq_num_patches, q=self.sq_num_patches)
        return x
    
    def unpatchify(self, sequence):
        return rearrange(sequence, "b (p q) c h w -> b c (p h) (q w)", p=self.sq_num_patches, q=self.sq_num_patches)
    
    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=(-1, -2, -3), keepdim=True)
            var = target.var(dim=(-1, -2, -3), keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=(-1, -2, -3))  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def merge_patches(self, true_patches, pred_patches, is_masked):
        true_patches = true_patches * (1 - is_masked[:, :, None, None, None])
        pred_patches = pred_patches * is_masked[:, :, None, None, None]
        mixed_image = self.unpatchify(true_patches + pred_patches)
        return mixed_image
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
    ) -> dict:
        
        if len(pixel_values.size())==3:
            pixel_values = pixel_values[:, None]

        outputs = self.vit(
            pixel_values,
            noise=noise,
        )

        latent = outputs["last_hidden_state"]
        ids_restore = outputs["ids_restore"]
        mask = outputs["mask"]

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs["logits"]  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        mixed_image = self.merge_patches(
            true_patches=self.patchify(pixel_values),
            pred_patches=logits,
            is_masked=mask
        )

        loss = self.forward_loss(pixel_values, logits, mask)

        return dict(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            mixed_image=mixed_image,
        )