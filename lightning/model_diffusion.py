import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.utils import make_grid


class DiffusionNetPL(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 betas: tuple[float, float],
                 img_size: int,
                 img_channels: int,
                 model_base_channels: int,
                 noise_steps: int,
                 num_classes: int=None,
                 num_monitors=16,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.num_classes = num_classes
        self.num_monitors = num_monitors
        self.model = UNet(
            img_size=img_size,
            c_base=model_base_channels,
            c_in=img_channels,
            c_out=img_channels,
            num_classes=num_classes,
        )
        self.diffusion = Diffusion(
            img_size=img_size,
            img_channels=img_channels,
            noise_steps=noise_steps
        )
        self.loss_fn = nn.MSELoss()
        
    def training_step(self, batch, batch_idx):
        # Load images and/or labels
        if isinstance(batch, tuple):
            imgs, lbls = batch
        else:
            imgs = batch
            lbls = None
        if np.random.random() < 0.1 or self.num_classes is None:
            lbls = None  # classifier-free guidance
        batch_size = imgs.shape[0]
        device = imgs.device
        
        # Sample noised time-steps, predict noise, compute loss
        t = self.diffusion.sample_timesteps(batch_size, device)
        x_t, noise = self.diffusion.noise_images(imgs, t)
        predicted_noise = self.model(x_t, t, lbls)
        loss = self.loss_fn(noise, predicted_noise)
        self.log_dict({"loss": loss})
        
        # Monitor progress
        if batch_idx > 0 and self.global_step * batch_size % 25_000 == 0:
            samples = self.diffusion.sample(
                self.model, n=self.num_monitors, labels=lbls
            )
            self._log_generated_images(samples=samples, step=self.global_step)
        
        # Return loss for pytorch lightning
        return loss
    
    def _log_generated_images(self, samples, step):
        """
        Log model reconstructions vs real images in tensorboard.
        """
        # samples = ((samples + 1.0) / 2.0).clamp(0.0, 1.0)  # see data transform
        samples = make_grid(samples[:self.num_monitors], nrow=4, normalize=True)
        self.logger.experiment.add_image("samples", samples, step)
            
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas
        )
        return optimizer
    

class UNet(nn.Module):
    def __init__(self,
                 img_size: int=64,
                 c_base: int=64,
                 c_in: int=3,
                 c_out: int=3,
                 time_dim: int=256,
                 num_classes: int=None,
    ) -> None:
        super().__init__()
        # Time-step and/or class embeddings
        self.time_dim = time_dim
        if num_classes is not None:
            self.class_embedding_layer = nn.Embedding(num_classes, time_dim)
        
        # Encoding
        self.inc = DoubleConv(c_in, c_base)
        self.down1 = Down(c_base, c_base * 2)
        self.sa1 = SelfAttention(c_base * 2)
        self.down2 = Down(c_base * 2, c_base * 4)
        self.sa2 = SelfAttention(c_base * 4)
        self.down3 = Down(c_base * 4, c_base * 4)
        self.sa3 = SelfAttention(c_base * 4)
        
        # Bottleneck
        self.bot1 = DoubleConv(c_base * 4, c_base * 8)
        self.bot2 = DoubleConv(c_base * 8, c_base * 8)
        self.bot3 = DoubleConv(c_base * 8, c_base * 4)
        
        # Decoding
        self.up1 = Up(c_base * 8, c_base * 2)  # includes skip-connections!
        self.sa4 = SelfAttention(c_base * 2)
        self.up2 = Up(c_base * 4, c_base)  # includes skip-connections!
        self.sa5 = SelfAttention(c_base)
        self.up3 = Up(c_base * 2, c_base)  # includes skip-connections!
        self.sa6 = SelfAttention(c_base)
        self.outc = nn.Conv2d(c_base, c_out, kernel_size=1)
    
    def _get_input_device(self):
        return self.inc._get_input_device()
    
    def pos_encoding(self, t: torch.Tensor, channels: int):
        exp = torch.arange(0, channels, 2, device=t.device).float() / channels
        inv_freq = 1.0 / (10_000 ** exp)  # 10 * time_steps?
        pos_enc_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim=1)
        return pos_enc
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: int=None):
        # Time-step and/or class embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t = t + self.class_embedding_layer(y)
        
        # Encoding
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        # Decoding
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class Diffusion:
    def __init__(self,
                 img_size=64,
                 img_channels=3,
                 noise_steps=100,  # 1000
                 beta_start=1e-4,
                 beta_end=0.02,
                 ) -> None:
        # Noising parameters
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        
        # Noise schedule arrays
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x: torch.Tensor, t: torch.Tensor):
        self.alpha_hat = self.alpha_hat.to(x.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        noised_imgs = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return noised_imgs, epsilon
    
    def sample_timesteps(self, n: int, device: str):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=device)
    
    def sample(self,
               model: nn.Module,
               n: int,
               labels: torch.Tensor,
               cfg_scale: int=3,
    ) -> torch.Tensor:
        # Initialize model and get correct input device
        model.eval()
        device = model._get_input_device()
        with torch.no_grad():
            
            # Generate pure standard noise and go back through the noise steps
            x = torch.randn(
                (n, self.img_channels, self.img_size, self.img_size),
                device=device,
            )
            steps = reversed(range(1, self.noise_steps))
            for i in tqdm(steps, position=0, desc='Sampling model generation'):
                
                # Predict noise from step i to step i - 1
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t, labels)
                
                # Classifier free guidance
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale,
                    )
                
                # Take correct noise schedule parameters
                alpha = self.alpha.to(device)[t][:, None, None, None]
                alpha_hat = self.alpha_hat.to(device)[t][:, None, None, None]
                beta = self.beta.to(device)[t][:, None, None, None]
                
                # Generate noise from step i - 1 to step i
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Compute one denoising step closer to the "original" image
                alpha_term = (1 - alpha) / torch.sqrt(1 - alpha_hat)
                x_term = (x - alpha_term * predicted_noise)
                noise_term = torch.sqrt(beta) * noise
                x = 1 / torch.sqrt(alpha) * x_term + noise_term
        
        # Put model back in train mode and return predicted image (noise step 0)
        model.train()
        return x
        

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
        
    def _get_input_device(self):
        return self.double_conv[0].weight.device
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.time_embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.maxpool_conv(x)
        emb = self.time_embedding_layer(t)[:, :, None, None]
        emb = emb.repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.time_embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    
    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.time_embedding_layer(t)[:, :, None, None]
        emb = emb.repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        
    def forward(self, x: torch.Tensor):
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, height * width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        attn = attn + x
        attn = self.ff_self(attn) + attn
        attn = attn.swapaxes(2, 1).view(batch_size, num_channels, height, width)
        return attn
    