import torch
import pytorch_lightning as pl
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from torchvision.utils import make_grid


class VQ_GAN_PL(pl.LightningModule):
    def __init__(
        self,
        lr,
        betas,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        num_monitors=16,
        ckpt_path=None,
        ignore_keys=[],
    ):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.num_monitors = num_monitors
        self.automatic_optimization = False
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx):
        # Running optimizer
        optimizer_idx = batch_idx % len(self.optimizers())
        running_optim = self.optimizers()[optimizer_idx]
        
        # Encode data and reconstruct it from closest codebook vectors
        batch_rec, qloss = self(batch)
        
        # Train generator (auto-encoder) or discriminator
        loss, _ = self.loss(
            qloss,
            batch,
            batch_rec,
            optimizer_idx,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        
        # Manual backpropagation
        running_optim.zero_grad()
        self.manual_backward(loss)  #, retain_graph=(optimizer_idx == 0))  # or?
        running_optim.step()
        
        # Log correct losses
        if optimizer_idx == 0:
            self.log_dict({"ae_loss": loss, "q_loss": qloss}, prog_bar=False)
        elif optimizer_idx == 1:
            self.log_dict({"disc_loss": loss, "q_loss": qloss}, prog_bar=False)
            
        # Monitor model quality
        step = self.global_step // len(self.optimizers())  # both increment global step
        if step % 100 == 0:
            self._log_generated_images(batch, batch_rec, step)

    def _log_generated_images(self, real, fake, step):
        """
        Log model reconstructions vs real images in tensorboard.
        """
        real = ((real + 1.0) / 2.0).clamp(0.0, 1.0)  # see data transform
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)  # see data transform
        grid_real = make_grid(real[:self.num_monitors], nrow=4)  # , normalize=True)
        grid_fake = make_grid(fake[:self.num_monitors], nrow=4)  # , normalize=True)
        self.logger.experiment.add_image("real", grid_real, step)
        self.logger.experiment.add_image("fake", grid_fake, step)

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=self.lr,
            betas=self.betas,
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=self.lr, betas=self.betas,
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
