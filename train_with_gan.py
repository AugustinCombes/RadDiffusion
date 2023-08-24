import torch.multiprocessing as mp

from models.MAE_GAN import MAE_GAN
from dicom_dataset import get_splitted_dataloaders
from utils import plot_images_from_batch, plot_global_comparison

import math
from tqdm import tqdm
from itertools import chain

import torch
from torch import nn
import numpy as np
import transformers
from transformers import get_cosine_schedule_with_warmup

import warnings
import os
warnings.filterwarnings("ignore")

do_global_reconstruction=False
device = "cuda:2"
run_name = 'runGan'

def main():
    os.mkdir(run_name)

    config = transformers.ViTMAEConfig(
        hidden_size = 256, #512
        num_hidden_layers = 6,
        num_attention_heads = 8,
        intermediate_size = 512, #1024
        
        image_size = 256,
        patch_size = 32,
        num_channels = 1,
        
        decoder_num_attention_heads = 8,
        decoder_hidden_size = 256,
        decoder_num_hidden_layers = 2,
        decoder_intermediate_size = 512,

        mask_ratio = 0.75,
    )

    model = MAE_GAN(config).to(device)
    print(f"Number of million parameters: {model.num_parameters()/1e6}")
    model.train()


    batchSize = 512
    epochs = 501
    warmup_epochs = 0
    base_lr = 5e-4 #1e-4
    weight_decay = 3e-5 #5e-2 #3e-5
    
    train_dl, valid_dl, _ = get_splitted_dataloaders("preprocessed_data/xcr_256", device, batchSize, num_workers=4)

    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    g_optimizer = torch.optim.Adam(chain(model.vit.parameters(), model.decoder.parameters()), lr=base_lr)
    d_optimizer = torch.optim.Adam(chain(model.vit.parameters(), model.discriminator.parameters()), lr=base_lr)

    n_sample = 180776 #len(train_dl.dataset) #speed-up pour 256

    n_steps_per_epoch = math.ceil(n_sample/batchSize)
    n_steps = n_steps_per_epoch * epochs
    n_warmup_steps = n_steps_per_epoch * warmup_epochs 

    # Define reference image & noise for visualisation 
    batch_ref = next(iter(train_dl))
    seq_length = (model.config.image_size ** 2) // (model.config.patch_size ** 2)
    noise_ref = torch.rand(batchSize, seq_length, device=model.device)

    g_scheduler = get_cosine_schedule_with_warmup(g_optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=n_steps, num_cycles=1.5)
    d_scheduler = get_cosine_schedule_with_warmup(d_optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=n_steps, num_cycles=1.5)
    step = 0
    for epoch in range(epochs):
        epoch_stats = {
            "L_2": [],
            "L_adv": [],
            "gamma": [],
            "L_gen": [],
            "L_dis": [],
        }

        for idx, batch in tqdm(enumerate(train_dl)):
            step +=1
            
            #Generator
            g_optimizer.zero_grad()
            result = model(batch)
            
            l2_loss = result["loss"]
            y, y_hat = model.forward_discriminator_loss(result["mixed_image"], result["mask"])
            adv_loss = - (
                y * torch.log(y_hat)
                ).mean()
            # adv_loss = -(y * torch.log(y_hat) + (1-y) * torch.log(1 - y_hat)).mean()

            gamma = model.compute_gamma(l2_loss, adv_loss)
            g_loss = l2_loss + gamma * adv_loss
            g_loss.backward()
            g_optimizer.step(), g_scheduler.step()

            if True:
                epoch_stats["L_2"].append(l2_loss.detach().item())
                epoch_stats["L_adv"].append(adv_loss.detach().item())
                epoch_stats["gamma"].append(gamma.detach().item())
                epoch_stats["L_gen"].append(g_loss.detach().item())

            #Discriminator
            d_optimizer.zero_grad()
            _, y_hat = model.forward_discriminator_loss(result["mixed_image"].detach(), result["mask"].detach())

            false_negative_loss = (1-y) * torch.log(y_hat)
            false_positive_loss = y * torch.log(1 - y_hat)
            d_loss = - (
                false_negative_loss / (1 - config.mask_ratio) + 
                false_positive_loss / config.mask_ratio
                ).mean()
            d_loss.backward()
            d_optimizer.step(), d_scheduler.step()

            if True:
                epoch_stats["L_dis"].append(d_loss.detach().item())

        epoch_stats = {k: np.array(v).mean() for k,v in epoch_stats.items()}
        print("epoch", epoch, "=", epoch_stats)

        with torch.no_grad():
            res = model(batch_ref, noise=noise_ref)["mixed_image"].squeeze().cpu()
            for ex in range(5):
                model.save_img(
                    model.get_nth_boxed_visualisation(result, ex), 
                    f"{run_name}/epoch_{epoch}_v{ex}.jpg"
                )

    import datetime
    torch.save(model.state_dict(), str(datetime.datetime.now()) + ".pth")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()