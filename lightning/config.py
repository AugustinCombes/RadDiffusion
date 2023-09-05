import os
import transformers


# Training hyperparameters
MODEL_TYPE = "VQ_GAN"  # MAE_GAN, VQ_GAN
LEARNING_RATE = 5e-5
OPTIM_BETAS = (0.5, 0.9)
BATCH_SIZE = 16  # 512
NUM_EPOCHS = 50

# Data
IMG_CHANNELS = 1
IMG_SHAPE = (256, 256)
NUM_SAMPLES = 723862  # 180776 <- Augustin
DATA_MAIN_DIR = os.path.join("/", "data", "dt_group", "xcr_256")
NUM_WORKERS = 4

# Computations
ACCELERATOR = "gpu"
DEVICES = [1]
PRECISION = 32  # 16-mixed

# Model specific (MAE_GAN)
VIT_CONFIG = transformers.ViTMAEConfig(
    hidden_size=256,  # 256
    num_hidden_layers=8,  # 6
    num_attention_heads=8,  # 8
    intermediate_size=1024,  # 512
    image_size=256,  # 256
    patch_size=16,  # 32
    num_channels=1,  # 1
    decoder_num_attention_heads=8,  # 8
    decoder_hidden_size=256,  # 256
    decoder_num_hidden_layers=2,  # 2
    decoder_intermediate_size=1024,  # 512
    mask_ratio=0.75,  # 0.75
)

# Model specific (VQ_GAN)
VQ_CONFIG = {
    'latent_dim': 256,
    'num_codebook_vectors': 1024,
    'beta': 0.25,
    'disc_start_step': 2000,
    'disc_factor_init': 1.0,
    'vgg_loss_factor': 1.0,
    'l1_loss_factor': 1.0,
    'vgg_url_map': {'vgg_lpips': 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1'},
    'vgg_ckpt_map': {'vgg_lpips': 'vgg.pth'},
}
GPT_CONFIG = {
    'pkeep': 0.5,
    'sos_token': 0,
    'last_vqgan_ckpt_path': os.path.join('logs', 'VQ_GAN', 'ckpt', 'vq_gan.pt'),
    'lr_gpt': 4.5e-6,
    'betas_gpt': (0.9, 0.95)
}
