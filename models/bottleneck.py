import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from copy import deepcopy
import math
from typing import Optional, Set, Tuple, Union

from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings, ViTMAEPatchEmbeddings, ViTMAEAttention, ViTMAEIntermediate, ViTMAEPreTrainedModel
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAESelfOutput, ViTMAEDecoderOutput, ViTMAEOutput, ViTMAEForPreTrainingOutput, ViTMAESelfAttention
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput
from transformers import ViTPreTrainedModel

from src.multi_focal_loss import FocalWithLogitsLoss

class ViTMAEBottleneckModelOutput(ModelOutput):
    """
    Class for ViTMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    lengths: list = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ViTMAEBottleneckLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths,
        head_mask = None,
        output_attentions: bool = False,
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViTMAE, layernorm is applied before self-attention
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        

        #cls token intra-study inter-images averaging
        cls = layer_output[:, :1, :]
        index = torch.tensor([idx for idx, l in enumerate(lengths) for _ in range(l)], device=cls.device)
        index = index[:, None, None].expand_as(cls)
        cls = torch.zeros(lengths.shape[0], *cls.shape[1:], device=cls.device).scatter_add_(0, index, cls) / lengths[:, None, None]
        index = index[:, 0, 0]
        cls = cls[index]
        layer_output = torch.cat([cls, layer_output[:, 1:, :]], dim=1)


        outputs = (layer_output,) + outputs

        return outputs
    
class ViTMAEBottleneckEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTMAEBottleneckLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths,
        head_mask = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, lengths, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
class ViTMAEBottleneckModel(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTMAEEmbeddings(config)
        self.encoder = ViTMAEBottleneckEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        lengths: Optional[torch.IntTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        if lengths is None:
            raise ValueError("You have to specify lengths")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)

        encoder_outputs = self.encoder(
            embedding_output,
            lengths,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEBottleneckModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            lengths=lengths,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
class ViTMAEBottleneckDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAEBottleneckLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        lengths,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    None,
                )
            else:
                layer_outputs = layer_module(hidden_states, lengths, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
class ViTMAEBottleneckForPreTraining(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = ViTMAEBottleneckModel(config)
        self.decoder = ViTMAEBottleneckDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

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

        # pixel_values = torch.stack([image for study in pixel_values for image in study], dim=0) #*

        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        lengths: Optional[torch.IntTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            noise=noise,
            lengths=lengths,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask
        lengths = outputs.lengths

        decoder_outputs = self.decoder(latent, lengths, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
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
    
class ViTBottleneckForImageClassification(ViTPreTrainedModel):
    def __init__(self, config, class_means=None, from_checkpoint=None) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.class_means = None
        self.vit = ViTMAEBottleneckModel(config)

        if from_checkpoint is not None:
            self.vit.load_state_dict(
                torch.load(from_checkpoint)
                )

        assert class_means is None or len(class_means) == self.num_labels, "Should pass class_weights with length n_label"
        if class_means is not None:
            self.class_means = torch.tensor(class_means)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        lengths: Optional[torch.IntTensor] = None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            lengths=lengths,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_cls = outputs['last_hidden_state'][:, :1, :]
        index = torch.tensor([idx for idx, l in enumerate(lengths) for _ in range(l)], device=last_cls.device)
        index = index[:, None, None].expand_as(last_cls)
        cls = torch.zeros(lengths.shape[0], *last_cls.shape[1:], device=last_cls.device).scatter_add_(0, index, last_cls) / lengths[:, None, None]
        cls = cls.squeeze()

        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            if self.class_means is None:
                loss_fct = nn.BCEWithLogitsLoss() 
            else:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=1/self.class_means.sqrt().to(logits.device))
                # loss_fct = nn.BCEWithLogitsLoss()
                # loss_fct = nn.BCEWithLogitsLoss(pos_weight=1/self.class_means.to(logits.device))
                # loss_fct = FocalWithLogitsLoss(pos_weight=1/self.class_means.to(logits.device))

            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return dict(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class DiffMAEBottleneckDecoder(nn.Module):
    def __init__(self, config, num_patches, strategy=None):
        super().__init__()

        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.num_patches = num_patches

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size

        self.decoder_patch_embeddings = ViTMAEPatchEmbeddings(decoder_config)

        self.strategy = strategy
        if strategy is None:
            self.decoder_layers = nn.ModuleList(
                [ViTMAEBottleneckLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
            )
        else:
            self.decoder_layers = nn.ModuleList(
                [ViTMAEBottleneckCrossLayer(decoder_config, strategy) for _ in range(config.decoder_num_hidden_layers // 2)] #n^2 -> n^2 / 2
            )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        lengths,
        ids_restore,
        noised_pixel_values,
        output_attentions=False,
        output_hidden_states=False,
    ):
        # embed visible tokens
        x = self.decoder_embed(hidden_states)
        # embed noised patches
        noised_tokens = self.decoder_patch_embeddings(noised_pixel_values)

        len_keep = int(self.num_patches * (1 - self.config.mask_ratio))
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        ids_noised = ids_shuffle[:, len_keep:]
        ids_keep = ids_shuffle[:, :len_keep]

        # inject shuffled noised tokens in sequence
        noised_tokens = torch.gather(
            noised_tokens, 
            dim=1, 
            index=ids_noised.unsqueeze(-1).repeat(1, 1, self.config.decoder_hidden_size)
            )
        x_ = torch.cat([x[:, 1:, :], noised_tokens], dim=1) # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # append cls token

        hidden_states = x + self.decoder_pos_embed
        
        if self.strategy is None:
            # apply Transformer layers (blocks)
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            for i, layer_module in enumerate(self.decoder_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, lengths=lengths, head_mask=None, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

        else:
            cls = hidden_states[:, :1, :]
            context_hidden_states = torch.gather(hidden_states[:, 1:, :], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
            hidden_states = torch.gather(hidden_states[:, 1:, :], dim=1, index=ids_noised.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
            # hidden_states = torch.cat([cls, hidden_states], dim=1) # append cls token
            context_hidden_states = torch.cat([cls, context_hidden_states], dim=1) # append cls token

            # apply Transformer layers (blocks)
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            for i, layer_module in enumerate(self.decoder_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, context_hidden_states, lengths=lengths, head_mask=None, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)
        
        # remove cls token
        if self.strategy is None:
            logits = logits[:, 1:, :] #else the cls token is in the context

        return dict(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class DiffMAEBottleneckForPreTraining(ViTMAEPreTrainedModel):
    def __init__(self, config, strategy):
        super().__init__(config)
        self.config = config

        self.vit = ViTMAEBottleneckModel(config)
        self.decoder = DiffMAEBottleneckDecoder(config, num_patches=self.vit.embeddings.num_patches, strategy=strategy)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

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

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noised_pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        lengths: Optional[torch.IntTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            noise=noise,
            lengths=lengths,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask
        lengths = outputs.lengths

        decoder_outputs = self.decoder(
            latent, 
            lengths=lengths, 
            ids_restore=ids_restore, 
            noised_pixel_values=noised_pixel_values
            )
        logits = decoder_outputs["logits"]  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        if self.decoder.strategy is not None:
            len_keep = int(self.decoder.num_patches * (1 - self.config.mask_ratio))
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_noised = ids_shuffle[:, len_keep:]

            sequence_masked = torch.gather(self.patchify(pixel_values), dim=1, index=ids_noised.unsqueeze(-1).repeat(1, 1, logits.size(-1)))
            loss = ((logits - sequence_masked) ** 2).mean()

        else:
            loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
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

class ViTMAESCrossAttentionModule(ViTMAESelfAttention):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self, hidden_states, context_hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(context_hidden_states))
        value_layer = self.transpose_for_scores(self.value(context_hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    
class ViTMAECrossAttention(ViTMAEAttention):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__(config)
        self.attention = ViTMAESCrossAttentionModule(config)
        self.output = ViTMAESelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, context_hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class ViTMAEBottleneckCrossLayer(nn.Module):
    def __init__(self, config, strategy) -> None:
        super().__init__()

        available_strategies = ['cross-self', 'cross']
        if strategy not in available_strategies:
            raise NotImplementedError(
                f'Strategy {strategy} is not implemented, available strategies are {str(available_strategies)}'
                )
        self.strategy = strategy

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.cross_attention = ViTMAECrossAttention(config)
        if self.strategy == 'cross-self':
            self.attention = ViTMAEAttention(config)
            self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_intermediate = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        lengths,
        head_mask = None,
        output_attentions: bool = False,
    ):
        #cross-attention
        cross_attention_outputs = self.cross_attention(
            self.layernorm_before(hidden_states),  # in ViTMAE, layernorm is applied before self-attention
            context_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        #self-attention
        if self.strategy == 'cross-self':
            self_attention_outputs = self.attention(
                self.layernorm_intermediate(hidden_states),  # in ViTMAE, layernorm is applied before self-attention
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]

            # second residual connection
            hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # third residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        
        # #cls token intra-study inter-images averaging
        # cls = layer_output[:, :1, :]
        # index = torch.tensor([idx for idx, l in enumerate(lengths) for _ in range(l)], device=cls.device)
        # index = index[:, None, None].expand_as(cls)
        # cls = torch.zeros(lengths.shape[0], *cls.shape[1:], device=cls.device).scatter_add_(0, index, cls) / lengths[:, None, None]
        # index = index[:, 0, 0]
        # cls = cls[index]
        # layer_output = torch.cat([cls, layer_output[:, 1:, :]], dim=1)


        outputs = (layer_output,) + outputs

        return outputs