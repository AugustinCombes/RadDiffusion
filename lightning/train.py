import config
import torch
import pytorch_lightning as pl
from model_vq_gan import VQ_GAN_PL
from model_mae_gan import MAE_GAN_PL
from model_diffusion import DiffusionNetPL
from dataset import CXRDataModule
from callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy


def main():
    torch.set_float32_matmul_precision("medium")
    model = select_model()
    data_module = CXRDataModule(
        data_dir=config.DATA_MAIN_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        strategy="auto",  # DDPStrategy() if len(config.DEVICES) > 1 else None
        logger=TensorBoardLogger("logs", name=config.MODEL_TYPE),
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        limit_train_batches=config.MAX_STEPS_PER_TRAIN_EPOCH,
        precision=config.PRECISION,
        log_every_n_steps=50,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)
    # trainer.validate(model, data_module)
    # trainer.test(model, data_module)


def select_model():
    if config.MODEL_TYPE == "MAE_GAN":
        model = MAE_GAN_PL(
            lr=config.LEARNING_RATE,
            betas=config.OPTIM_BETAS,
            vit_config=config.VIT_CONFIG,
        )
    elif config.MODEL_TYPE == "VQ_GAN":
        model = VQ_GAN_PL(
            lr=config.LEARNING_RATE,
            betas=config.OPTIM_BETAS,
            **config.VQ_GAN_CONFIG,
        )
    elif config.MODEL_TYPE == "DIFF":
        model = DiffusionNetPL(
            lr=config.LEARNING_RATE,
            betas=config.OPTIM_BETAS,
            img_size=config.IMG_SHAPE[0],
            img_channels=config.IMG_CHANNELS,
            **config.DIFF_CONFIG,
        )
    else:
        raise ValueError("Invalid model type selected.")
    return model


if __name__ == "__main__":
    main()
