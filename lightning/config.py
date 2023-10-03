import os
import transformers


# Training hyperparameters
MODEL_TYPE = "DIFF"  # MAE_GAN, VQ_GAN, DIFF
LEARNING_RATE = {
    "MAE_GAN": 1e-4,
    "VQ_GAN": 4.5e-6,
    "DIFF": 3e-4,
}[MODEL_TYPE]
OPTIM_BETAS = {
    "MAE_GAN": (0.5, 0.9),
    "VQ_GAN": (0.5, 0.9),
    "DIFF": (0.9, 0.999),
}[MODEL_TYPE]
BATCH_SIZE = {
    "MAE_GAN": 512,
    "VQ_GAN": 12,
    "DIFF": 1,
}[MODEL_TYPE]
NUM_EPOCHS = {
    "MAE_GAN": 10,
    "VQ_GAN": 10,
    "DIFF": 100,
}[MODEL_TYPE]
MAX_STEPS_PER_TRAIN_EPOCH = None  # int or None for using full dataset


# Data
IMG_CHANNELS = 1
IMG_SHAPE = (128, 128)  # (256, 256)
NUM_SAMPLES = 298648  # 180776 <- img_ids[0] only
DATA_MAIN_DIR = os.path.join("/", "data", "dt_group", "xcr_256")
NUM_WORKERS = 4


# Computations
ACCELERATOR = "gpu"
DEVICES = [1]
PRECISION = 32  # 16-mixed


# Model specific (MAE_GAN)
VIT_CONFIG_DICT = {
  "attention_probs_dropout_prob": 0.0,
  "decoder_hidden_size": 384,
  "decoder_intermediate_size": 1536,
  "decoder_num_attention_heads": 12,
  "decoder_num_hidden_layers": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 768,
  "image_size": 256,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "mask_ratio": 0.75,
  "model_type": "vit_mae",
  "norm_pix_loss": True,
  "num_attention_heads": 12,
  "num_channels": 1,
  "num_hidden_layers": 12,
  "patch_size": 32,
  "qkv_bias": True,
  "transformers_version": "4.30.2"
}
VIT_CONFIG = transformers.ViTMAEConfig(**VIT_CONFIG_DICT)


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


# Model specific (DIFF)
assert IMG_SHAPE[0] == IMG_SHAPE[1]
DIFF_CONFIG = {
    "model_base_channels": 64,
    "noise_steps": 1000,
    "num_classes": None,
}
