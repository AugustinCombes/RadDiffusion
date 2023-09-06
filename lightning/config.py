import os
import transformers


# Training hyperparameters
MODEL_TYPE = "VQ_GAN"  # MAE_GAN, VQ_GAN
LEARNING_RATE = 4.5e-6
OPTIM_BETAS = (0.5, 0.9)
BATCH_SIZE = 12  # 512
NUM_EPOCHS = 10
MAX_STEPS_PER_TRAIN_EPOCH = None  # int or None for using full dataset

# Data
IMG_CHANNELS = 1
IMG_SHAPE = (256, 256)
NUM_SAMPLES = 298648  # 180776 <- img_ids[0] only
DATA_MAIN_DIR = os.path.join("/", "data", "dt_group", "xcr_256")
NUM_WORKERS = 8

# Computations
ACCELERATOR = "gpu"
DEVICES = [1]
PRECISION = 32  # 16-mixed

# Model specific (MAE_GAN)
VIT_CONFIG = transformers.ViTMAEConfig(
    attention_probs_dropout_prob=0.0,
    decoder_hidden_size=192,
    decoder_intermediate_size=768,
    decoder_num_attention_heads=12,
    decoder_num_hidden_layers=2,
    hidden_act="gelu",
    hidden_dropout_prob=0.3,
    hidden_size=384,
    image_size=256,
    initializer_range=0.02,
    intermediate_size=1536,
    layer_norm_eps=1e-12,
    mask_ratio=0.75,
    model_type="vit_mae",
    norm_pix_loss=True,
    num_attention_heads=12,
    num_channels=1,
    num_hidden_layers=12,
    patch_size=32,
    qkv_bias=True,
    transformers_version="4.30.2",
)

# Model specific (VQ_GAN)
VQ_GAN_CONFIG = {
    "embed_dim": 256,
    "n_embed": 1024,
    "ddconfig": {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 1,  # 3 (?) <- pas utilisé dans leur implémentation
        "out_ch": 1,  # 3
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
    },
    "lossconfig": {
        "disc_conditional": False,
        "disc_in_channels": 1,  # 3
        "disc_start": 10001,  # 250001,
        "disc_weight": 0.8,
        "codebook_weight": 1.0,
    },
}
