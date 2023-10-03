import config
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from lightning_gpt.models import MinGPT
from tqdm import tqdm
from itertools import chain
from collections import namedtuple
from torchvision.utils import make_grid
from torchvision.models import vgg16, VGG16_Weights


class VQ_GAN_PL(pl.LightningModule):
    """
    Pytorch-lightning module to train the MAE-GAN of Fei et al. (2023)
    """

    def __init__(self, lr, betas, vq_config, num_monitors=16):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.vq_config = vq_config
        self.num_monitors = num_monitors  # how many images are monitored
        self.automatic_optimization = False  # necessary to have 2 optimizers
        self.steps_per_epoch = config.NUM_SAMPLES // config.BATCH_SIZE

        self.vqgan = VQGAN(img_channels=config.IMG_CHANNELS, **vq_config)
        self.discriminator = Discriminator(config.IMG_CHANNELS)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS_Loss(vq_config).eval()

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Define what happens during one batch when training the model.
        """
        # Load images and generate reconstructions
        decoded_batch, _, q_loss = self.vqgan(batch)

        # Judge is real and fake are real with discriminator
        disc_real = self.discriminator(batch)
        disc_fake = self.discriminator(decoded_batch)

        # Make sure discriminator does not learn too early
        disc_factor = self._adopt_vqgan_weight(
            disc_factor=self.vq_config["disc_factor_init"],
            threshold=self.vq_config["disc_start_step"],
            batch_idx=batch_idx,
        )

        # Feature-based losses (pixels and vgg-features)
        vgg_loss = self.perceptual_loss(batch, decoded_batch)
        vgg_loss = vgg_loss * self.vq_config["vgg_loss_factor"]
        l1_loss = torch.abs(batch - decoded_batch)
        l1_loss = l1_loss * self.vq_config["l1_loss_factor"]
        feature_loss = (vgg_loss + l1_loss).mean()

        # Pure generator loss
        gen_loss = -torch.mean(disc_fake)

        # Combine both "reconstruction" losses
        lambda_ = self.vqgan.calculate_lambda(feature_loss, gen_loss)
        vq_loss = (
            feature_loss
            + q_loss  # image-features vs decoded-features
            + disc_factor * lambda_ * gen_loss  # code-book loss  # gan-like loss
        )

        # Now compute discriminator loss (plus SVM relu trick)
        d_loss_real = torch.mean(F.relu(1.0 - disc_real))
        d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        disc_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        # Update parameters (in correct order!)
        opt_vq, opt_disc = self.optimizers()
        opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_vq.step()
        opt_disc.step()

        # Log loss info
        self.log_dict(
            {
                "q_loss": q_loss,
                "feature_loss": feature_loss,
                "gen_loss": gen_loss,
                "vq_loss": vq_loss,
                "disc_loss": disc_loss,
            },
            prog_bar=False,
        )

        # Monitor model quality
        step = self.global_step // len(self.optimizers())  # both increment global step
        if step % 100 == 0:
            self._log_generated_images(real=batch, fake=decoded_batch, step=step)

    def _log_generated_images(self, real, fake, step):
        """
        Log model reconstructions vs real images in tensorboard.
        """
        real = (real + 1.0) / 2.0  # see data transform
        fake = (fake + 1.0) / 2.0  # see data transform
        grid_real = make_grid(real[: self.num_monitors], nrow=4)  # , normalize=True)
        grid_fake = make_grid(fake[: self.num_monitors], nrow=4)  # , normalize=True)
        self.logger.experiment.add_image("real", grid_real, step)
        self.logger.experiment.add_image("fake", grid_fake, step)

    def _adopt_vqgan_weight(self, disc_factor, threshold, batch_idx, value=0.0):
        """
        Compute factor to make sure the discriminator does not learn too early
        """
        global_batch_idx = self.current_epoch * self.steps_per_epoch + batch_idx
        if global_batch_idx < threshold:
            disc_factor = value
        return disc_factor  # used to start discriminator training only later

    def configure_optimizers(self):
        """
        Define optimizers and schedulers for the different parts of the model.
        """
        opt_vq = optim.Adam(
            chain(
                self.vqgan.encoder.parameters(),
                self.vqgan.decoder.parameters(),
                self.vqgan.codebook.parameters(),
                self.vqgan.pre_quant_conv.parameters(),
                self.vqgan.post_quant_conv.parameters(),
            ),
            lr=self.lr,
            eps=1e-08,
            betas=self.betas,
        )
        opt_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            eps=1e-08,
            betas=self.betas,
        )
        return [opt_vq, opt_disc], []


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_out_channels_same = in_channels == out_channels

        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if not self.in_out_channels_same:
            self.channel_adapter = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_out_channels_same:
            return x + self.block(x)
        else:
            return self.channel_adapter(x) + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)  # for perfect size-match
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        # Get values
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # permute is not inplace!
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        # Compute attention
        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        # Return attended with skip connection
        attnded = torch.bmm(v, attn)
        attnded = attnded.reshape(b, c, h, w)
        return x + attnded


class Encoder(nn.Module):
    def __init__(self, img_channels, latent_dim):
        super().__init__()

        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        resolution = 256  # initial resolution
        num_residual_blocks = 2

        layers = [nn.Conv2d(img_channels, channels[0], 3, 1, 1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels  # effect only first time

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != len(channels) - 2:
                layers.append(DownsampleBlock(out_channels))  # channels[i + 1]
                resolution // 2

        layers.extend(
            [
                ResidualBlock(out_channels, out_channels),  # i.e., channels[-1]
                NonLocalBlock(out_channels),
                ResidualBlock(out_channels, out_channels),
                GroupNorm(out_channels),
                Swish(),
                nn.Conv2d(out_channels, latent_dim, 3, 1, 1),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, img_channels, latent_dim):
        super().__init__()

        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        resolution = 16  # initial resolution
        num_residual_blocks = 3

        in_channels = channels[0]
        layers = [
            nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i in range(len(channels)):
            out_channels = channels[i]

            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels  # effect only first time

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != 0:
                layers.append(UpsampleBlock(in_channels))
                resolution *= 2

        layers.extend(
            [
                GroupNorm(in_channels),
                Swish(),
                nn.Conv2d(in_channels, img_channels, 3, 1, 1),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CodeBook(nn.Module):
    def __init__(self, latent_dim, num_codebook_vectors, beta):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_codebook_vectors = num_codebook_vectors
        self.beta = beta

        self.embedding = nn.Embedding(num_codebook_vectors, latent_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_codebook_vectors,
            1.0 / num_codebook_vectors,
        )

    def forward(self, z):
        # Reshape latent space for direct comparison with code book
        z = z.permute(0, 2, 3, 1).contiguous()  # channels last
        z_flattened = z.view(-1, self.latent_dim)

        # Commpute L2 distance in an extended way (to keep dims correct)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.sum(torch.matmul(z_flattened, self.embedding.weight.t()))
        )

        # Find, for each z vector, the closest element in the code book
        min_dist_indices = torch.argmin(d, dim=1)
        z_q = z_flattened[min_dist_indices].view(z.shape)

        # Compute codebook loss, taking care of gradient flow
        loss = (
            torch.mean((z_q.detach() - z) ** 2)
            + torch.mean((z_q - z.detach()) ** 2) * self.beta
        )
        z_q = z + (z_q - z).detach()  # trick to keep gradient flowing through z_q!

        # Return z_q with original shape of z, as well as its indices and loss
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q, min_dist_indices, loss  # min dist indices used by mini-GPT


class VQGAN(nn.Module):
    def __init__(
        self,
        img_channels,
        latent_dim,
        num_codebook_vectors,
        beta,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)
        self.codebook = CodeBook(latent_dim, num_codebook_vectors, beta)
        self.pre_quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        pre_conved = self.pre_quant_conv(encoded)
        codebook_mapped, codebook_indices, q_loss = self.codebook(pre_conved)
        post_conved = self.post_quant_conv(codebook_mapped)
        decoded = self.decoder(post_conved)
        return decoded, codebook_indices, q_loss

    def encode(self, x):
        encoded = self.encoder(x)
        pre_conved = self.pre_quant_conv(encoded)
        codebook_mapped, codebook_indices, q_loss = self.codebook(pre_conved)
        return codebook_mapped, codebook_indices, q_loss

    def decode(self, z):
        post_conved = self.post_quant_conv(z)
        decoded = self.decoder(post_conved)
        return decoded

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weights = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(
            outputs=perceptual_loss,
            inputs=last_layer_weights,
            retain_graph=True,
        )[0]
        gan_loss_grads = torch.autograd.grad(
            outputs=gan_loss,
            inputs=last_layer_weights,
            retain_graph=True,
        )[0]
        lambda_ = torch.norm(perceptual_loss_grads) / (
            torch.norm(gan_loss_grads) + 1e-4
        )
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()
        return 0.8 * lambda_  # why 0.8??

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


class Discriminator(nn.Module):
    def __init__(self, img_channels, last_channels=64, num_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(img_channels, last_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]
        channel_mult = 1  # initial value

        for i in range(1, num_layers + 1):
            channel_mult_last = channel_mult
            channel_mult = min(2**i, 8)

            in_channels = last_channels * channel_mult_last
            out_channels = last_channels * channel_mult
            stride = 2 if i < num_layers else 1

            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        layers.append(nn.Conv2d(out_channels, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS_Loss(nn.Module):
    def __init__(self, vq_config):
        super(LPIPS_Loss, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()
        self.lins = nn.ModuleList(
            [
                NetLinLayer(self.channels[0]),
                NetLinLayer(self.channels[1]),
                NetLinLayer(self.channels[2]),
                NetLinLayer(self.channels[3]),
                NetLinLayer(self.channels[4]),
            ]
        )

        self.load_from_pretrained(vq_config)
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, vq_config, name="vgg_lpips"):
        url_map = vq_config["vgg_url_map"]
        ckpt_map = vq_config["vgg_ckpt_map"]
        ckpt = get_vgg_ckpt_path(name, "vgg_lpips", url_map, ckpt_map)
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )

    def forward(self, real_x, fake_x):
        features_real = self.vgg(self.scaling_layer(real_x))
        features_fake = self.vgg(self.scaling_layer(fake_x))

        diffs = {}
        for i in range(len(self.channels)):
            diffs[i] = (
                norm_tensor(features_real[i]) - norm_tensor(features_fake[i])
            ) ** 2

        return sum(
            [
                spatial_average(self.lins[i].model(diffs[i]))
                for i in range(len(self.channels))
            ]
        )


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, x):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(), nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(VGG16_Weights.IMAGENET1K_V1).features
        slices = [vgg_pretrained_features[i] for i in range(30)]
        self.slice1 = nn.Sequential(*slices[0:4])
        self.slice2 = nn.Sequential(*slices[4:9])
        self.slice3 = nn.Sequential(*slices[9:16])
        self.slice4 = nn.Sequential(*slices[16:23])
        self.slice5 = nn.Sequential(*slices[23:30])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        vgg_outputs = namedtuple(
            "VGGOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def norm_tensor(x):
    """
    Normalize images by their length to make them unit vector?
    :param x: batch of images
    :return: normalized batch of images
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)


def spatial_average(x):
    """
    Images have: batch_size x c x w x h --> average over width and height channel
    :param x: batch of images
    :return: averaged images along width and height
    """
    return x.mean([2, 3], keepdim=True)


def download_vgg(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def get_vgg_ckpt_path(name, root, url_map, ckpt_map):
    assert name in url_map
    path = os.path.join(root, ckpt_map[name])
    if not os.path.exists(path):
        print(f"Downloading {name} model from {url_map[name]} to {path}")
        download_vgg(url_map[name], path)
    return path


class VQGANGPT(nn.Module):
    def __init__(
        self,
        device,
        ckpt_path,
        img_channels,
        latent_dim,
        num_codebook_vectors,
        beta,
        sos_token,
        pkeep,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.sos_token = sos_token
        self.pkeep = pkeep
        self.transformer = MinGPT(  # NanoGPT?
            vocab_size=num_codebook_vectors,
            block_size=512,
            n_layers=24,
            n_heads=12,
            n_embd=1024,
        )
        self.vqgan = self.load_vqgan(
            ckpt_path,
            img_channels,
            latent_dim,
            num_codebook_vectors,
            beta,
            device,
        )

    @staticmethod
    def load_vqgan(
        ckpt_path,
        img_channels,
        latent_dim,
        num_codebook_vectors,
        beta,
        device,
    ):
        model = VQGAN(img_channels, latent_dim, num_codebook_vectors, beta, device)
        model.load_checkpoint(ckpt_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def image_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        vectors = self.vqgan.codebook.embedding(indices)
        vectors = vectors.reshape(indices.shape[0], p1, p2, 256)  # why hardcode
        vectors = vectors.permute(0, 3, 1, 2)  # check codebook code

        images = self.vqgan.decode(vectors)
        return images.cpu().detach().numpy()[0].transpose(1, 2, 0)

    def forward(self, x):
        # Load data (images) and encode
        _, indices = self.image_to_z(x)

        # Define start-of-sentence token and random masking indices
        sos_token_batch = torch.ones(x.shape[0], 1) * self.sos_token  # repeat??
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)

        # Generate random indices at mask locations
        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  # -> random replace mask for 50% of vectors
        mask = mask.round().to(dtype=torch.int64)  # why round()?
        new_indices = torch.cat(
            [
                sos_token_batch,
                mask * indices + (1 - mask) * random_indices,
            ],
            dim=1,
        )

        # Keep track of original indices and generate reconstruction
        target = indices
        logits, _ = self.transformer(new_indices[:, :-1])  # last one not used
        return logits, target

    def top_k_logits(self, logits, top_k):
        # Function used for sampling images
        v, _ = torch.topk(logits, top_k)  # dim = -1 default
        out = logits.clone()  # avoid modifying logits
        out[out < v[..., [-1]]] = -float(
            "inf"
        )  # rien compris mais set les pas top-k to a very low value
        return out

    @torch.no_grad()
    def sample(self, x, sos_token, steps, temperature=1.0, top_k=100):
        # Prepare transformer and prepend start-of-sequence token
        self.transformer.eval()
        x = torch.cat((sos_token, x), dim=1)

        # Generate tokens one by one
        for _ in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature  # pas compris

            # Select top-k logits (set others to very low prob value) <- seems super strong way
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            # Generate next token batch using logits
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_sample=1)
            x = torch.cat([x, next_idx], dim=1)

        # Return final output, after removing the start-of-sequence token
        x = x[:, sos_token.shape[1] :]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        # Encode image and initialize start-of-sequence token batch
        _, indices = self.image_to_z(x)
        sos_token_batch = torch.ones(x.shape[0], 1) * self.sos_token  # repeat??
        sos_token_batch = sos_token_batch.long().to("cuda")  # why so specific??

        # Generate reconstruction from half the indices the encoded image
        start_indices = indices[:, : indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices,
            sos_token_batch,
            step=indices.shape[1] - start_indices.shape[1],
        )
        half_sample = self.z_to_image(sample_indices)

        # Generate reconstruction from nothing, using only the GPT model
        start_indices = indices[:, :0]  # for correct shape
        sample_indices = self.sample(
            start_indices,
            sos_token_batch,
            steps=indices.shape[1],
        )
        full_sample = self.z_to_image(sample_indices)

        # Generate reconstruction with as the normal vqgan model would do
        x_rec = self.z_to_image(indices)

        # Report the different results
        output = dict()
        output["input"] = x
        output["rec"] = x_rec
        output["half_sample"] = half_sample
        output["full_sample"] = full_sample
        return output
