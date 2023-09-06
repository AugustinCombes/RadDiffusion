import config
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEModelOutput,
)
from transformers import ViTMAELayer, ViTMAEConfig
from itertools import chain
from torchvision.utils import make_grid


class MAE_GAN_PL(pl.LightningModule):
    """
    Pytorch-lightning module to train the MAE-GAN of Fei et al. (2023)
    """

    def __init__(self, lr, betas, vit_config, num_monitors=16):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.num_monitors = num_monitors  # how many images are monitored
        self.automatic_optimization = False  # necessary to have 2 optimizers
        self.mask_ratio = vit_config.mask_ratio

        self.criterion = nn.BCEWithLogitsLoss()
        self.mae = ViTMAEForPreTrainingAndGeneration(vit_config)
        self.disc = ViTDiscriminator(vit_config)

    def _merge_patches(
        self,
        real_patches: torch.Tensor,  # shape (batch_size, num_patches, patch_size ** 2 * c)
        pred_patches: torch.Tensor,  # shape (batch_size, num_patches, patch_size ** 2 * c)
        mask: torch.Tensor,  # shape (batch_size, num_patches)
    ) -> torch.Tensor:  # shape (batch_size, c, h, w)
        """
        Mix predicted masked patches to real non-masked patches
        """
        real_patches = real_patches * (1 - mask[:, :, None])
        pred_patches = pred_patches * mask[:, :, None]
        mixed_image = self.mae.unpatchify(real_patches + pred_patches)
        return mixed_image

    def _compute_gamma(
        self,
        loss_pix: torch.Tensor,
        loss_adv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Taken from https://github.com/dome272/VQGAN-pytorch/blob/main/vqgan.py
        """
        last_layer_weight = self.mae.decoder.decoder_pred.weight
        l2_loss_grads = torch.autograd.grad(
            loss_pix,
            last_layer_weight,
            retain_graph=True,
        )[0]
        adv_loss_grads = torch.autograd.grad(
            loss_adv,
            last_layer_weight,
            retain_graph=True,
        )[0]
        gamma = torch.norm(l2_loss_grads) / (torch.norm(adv_loss_grads) + 1e-4)
        gamma = torch.clamp(gamma, 0, 1e4).detach()
        return gamma

    def _discriminate(
        self,
        images: torch.Tensor,  # shape (batch_size, c, h, w)
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Take in mixed image made of both reconstructed and original patches,
        returns a prediction of whether it is original (?), for each patch.
        """
        vit_output = self.mae.encode_without_mask(images, *args, **kwargs)
        encoded_patches = vit_output.last_hidden_state[:, 1:]  # no [CLS] token
        disc_pred = self.disc(encoded_patches)  # decode all patches
        return disc_pred.squeeze()  # (batch_size, n_patches)

    def _reconstruct(
        self,
        images: torch.Tensor,  # shape (batch_size, c, h, w)
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Take in an image, cut it into patches, mask a given ratio of patches,
        encode non-masked patches, uses encoding to reconstruct masked patches.
        Compute loss based on the l2-norm of masked vs reconstructed patches.
        """
        # Encode image as a sequence of vit-embeddings (only the ones not masked)
        vit_output = self.mae.vit(pixel_values=images, *args, **kwargs)
        last_vit_hidden_state = vit_output.last_hidden_state  # keep [CLS] token
        vit_ids_restore = vit_output.ids_restore
        vit_mask = vit_output.mask

        # Decode image patches from not-masked vit-embeddings
        decoder_output = self.mae.decoder(last_vit_hidden_state, vit_ids_restore)
        reconstructed = decoder_output.logits

        # New image from original unmasked patches and reconstructed masked ones
        mixed_images = self._merge_patches(
            real_patches=self.mae.patchify(images),
            pred_patches=reconstructed,
            mask=vit_mask,
        )

        # Compute basic pixel reconstruction loss from reconstructed patch
        loss_pix = self.mae.forward_loss(images, reconstructed, vit_output.mask)

        return {
            "mixed_images": mixed_images,
            "loss_pix": loss_pix,
            "mask": vit_mask,
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Define what happens during one batch when training the model.
        """
        # Load optimizers, run model and generate reconstruction ("fake")
        optim_gen, optim_disc = self.optimizers()
        mae_output = self._reconstruct(batch)
        mixed_images = mae_output["mixed_images"]
        mask = mae_output["mask"]

        # Compute generator loss (wants fakes to be classfied as 1 by disc)
        disc_output_gen = self._discriminate(mixed_images)
        disc_fake_gen = disc_output_gen[mask == 1]
        loss_adv = self.criterion(disc_fake_gen, torch.ones_like(disc_fake_gen))
        loss_pix = mae_output["loss_pix"]
        gamma = self._compute_gamma(loss_pix, loss_adv)
        loss_gen = loss_pix + gamma * loss_adv

        # Backpropagate generator loss
        optim_gen.zero_grad()
        self.manual_backward(loss_gen, retain_graph=True)
        optim_gen.step()

        # Compute discriminator loss (wants reals -> 1; fakes -> 0)
        disc_output = self._discriminate(mixed_images)
        disc_real = disc_output[mask == 0]
        disc_fake = disc_output[mask == 1]
        loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = loss_disc_real + loss_disc_fake  # weight with mask_ratio?

        # Backpropagate discriminator loss
        optim_disc.zero_grad()
        self.manual_backward(loss_disc)
        optim_disc.step()

        # Call scheduler steps (because we are in manual_optimization mode)
        sched_gen, sched_disc = self.lr_schedulers()
        sched_gen.step()
        sched_disc.step()

        # Log loss info
        self.log_dict(
            {
                "loss_pix": loss_pix,
                "loss_adv": loss_adv,
                "gamma": gamma,
                "loss_gen": loss_gen,
                "loss_disc": loss_disc,
            },
            prog_bar=False,
        )

        # Monitor model quality
        step = self.global_step // len(self.optimizers())  # both increment global step
        if step % 100 == 0:
            self._log_generated_images(batch, mae_output["mixed_images"], step)

    def _log_generated_images(self, real, fake, step):
        """
        Log model reconstructions vs real images in tensorboard.
        """
        real = ((real + 1.0) / 2.0).clamp(0.0, 1.0)  # see data transform
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)  # see data transform
        grid_real = make_grid(real[: self.num_monitors], nrow=4)  # , normalize=True)
        grid_fake = make_grid(fake[: self.num_monitors], nrow=4)  # , normalize=True)
        self.logger.experiment.add_image("real", grid_real, step)
        self.logger.experiment.add_image("fake", grid_fake, step)

    def configure_optimizers(self):
        """
        Define optimizers and schedulers for the different parts of the model.
        """
        # Optimizers
        gen_params = chain(self.mae.vit.parameters(), self.mae.decoder.parameters())
        optim_gen = optim.Adam(gen_params, lr=self.lr, betas=self.betas)
        disc_params = chain(self.mae.vit.parameters(), self.disc.parameters())
        optim_disc = optim.Adam(disc_params, lr=self.lr, betas=self.betas)

        # Schedulers
        sched_params = {
            "max_lr": self.lr,
            "pct_start": 0.1,
            "epochs": config.NUM_EPOCHS + 1,  # avoid throwing error at the end
            "steps_per_epoch": config.NUM_SAMPLES // config.BATCH_SIZE,
        }
        sched_gen = optim.lr_scheduler.OneCycleLR(optim_gen, **sched_params)
        sched_disc = optim.lr_scheduler.OneCycleLR(optim_disc, **sched_params)

        # Return everything in the correct order
        optims = [optim_gen, optim_disc]
        scheds = [
            {"scheduler": sched_gen, "interval": "step"},
            {"scheduler": sched_disc, "interval": "step"},
        ]
        return optims, scheds


class ViTDiscriminator(nn.Module):
    """
    Classify image patches are real or reconstructed (by the generator).
    """

    class ViTMAELayerWrapper(nn.Module):
        """
        Helper class to use a ViTMAELayer inside a sequential module
        """

        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            output_tuple = self.layer(x)
            return output_tuple[0]

    def __init__(self, config):
        super().__init__()
        disc_config = ViTMAEConfig(
            hidden_size=config.decoder_hidden_size,
            num_hidden_layers=config.decoder_num_hidden_layers,
            num_attention_heads=config.decoder_num_attention_heads,
            intermediate_size=config.decoder_intermediate_size,
        )
        self.model = nn.Sequential(
            nn.Linear(config.hidden_size, disc_config.hidden_size),
            self.ViTMAELayerWrapper(ViTMAELayer(disc_config)),
            nn.LayerNorm(disc_config.hidden_size, disc_config.layer_norm_eps),
            nn.Linear(disc_config.hidden_size, 1),
            # nn.Sigmoid(),  # replaced by nn.BCEWithLogitsLoss as loss function
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # shape (batch_size, seq_len, hidden_size)
    ) -> torch.Tensor:
        return self.model(embeddings)


class ViTMAEForPreTrainingAndGeneration(ViTMAEForPreTraining):
    """
    Like VITMAEForPreTraining, but also to encode images with no mask
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__(config)
        self.original_mask_ratio = config.mask_ratio

    def encode_without_mask(self, image: torch.Tensor) -> ViTMAEModelOutput:
        """
        Vieille technique de brise-carre
        """
        self.vit.embeddings.config.mask_ratio = 0.0
        encoded = self.vit(image)
        self.vit.embeddings.config.mask_ratio = self.original_mask_ratio
        return encoded
