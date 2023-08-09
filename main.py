import torch.multiprocessing as mp

from models.InViT import InViT
from dicom_dataset import get_splitted_dataloaders
from utils import plot_images_from_batch, plot_global_comparison

import math
from tqdm import tqdm

import torch
from torch import nn
import numpy as np
import transformers
from transformers import get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

do_global_reconstruction=False
device = "cuda:2"
run_name = 'run2'

def main():
    config = transformers.ViTConfig(
        architectures= ["ViTModel"],
        attention_probs_dropout_prob= 0.0,
        encoder_stride= 16,
        hidden_act= "gelu",
        hidden_dropout_prob= 0.0,
        hidden_size= 256,
        image_size= 256,
        initializer_range= 0.02,
        intermediate_size= 512,
        layer_norm_eps= 1e-12,
        model_type= "vit",
        num_attention_heads= 8,
        num_channels= 1,
        num_hidden_layers= 6,
        patch_size= 16,
        qkv_bias= True,
        transformers_version= "4.30.2"
    )

    model = InViT(config, 
                  do_global_reconstruction=do_global_reconstruction, 
                  do_patchwise_deconvolution=True
                  ).to(device)
    print(f"Number of million parameters: {model.num_parameters()/1e6}")
    model.train()


    batchSize = 256
    epochs = 501
    warmup_epochs = 100
    base_lr = 1e-4
    weight_decay = 3e-5
    mask_ratio = 0.7
    num_patches = (model.config.image_size // model.config.patch_size) ** 2
    patch_width = model.config.image_size // model.config.patch_size

    train_dl, valid_dl, _ = get_splitted_dataloaders("preprocessed_data/xcr_256", device, batchSize)

    optimizer = torch.optim.Adam(model.parameters(), lr = base_lr, weight_decay = weight_decay)

    # n_sample = len(train_dl.dataset)
    # n_sample = 180402 #len(train_dl.dataset) #speed-up pour 1024
    # n_sample = 180776 #len(train_dl.dataset) #speed-up pour 512
    n_sample = 180776 #len(train_dl.dataset) #speed-up pour 256
    # n_sample = 179663 #len(train_dl.dataset) #speed-up pour 224

    n_steps_per_epoch = math.ceil(n_sample/batchSize)
    n_steps = n_steps_per_epoch * epochs
    n_warmup_steps = n_steps_per_epoch * warmup_epochs 

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=n_steps, num_cycles=1.5)


    step = 0
    for epoch in range(epochs):
        epoch_loss = []

        for idx, batch in tqdm(enumerate(train_dl)):
            step +=1
            optimizer.zero_grad()

            #generate mask
            image = batch['image'][:, None, :, :]#.to(device)
            mask = (torch.zeros((image.shape[0], num_patches)).float().uniform_(0, 1) < mask_ratio).int()
            mask = mask.to(device)

            result = model(image, bool_masked_pos=mask)
            
            patch_loss = result["patchwise_loss"]
            if do_global_reconstruction:
                global_loss = result["global_reconstruction_loss"]
                mu = max(0, (step - n_warmup_steps)/n_steps)
                weighted_loss = mu * global_loss + (1-mu) * patch_loss
            else:
                weighted_loss = patch_loss

            weighted_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss.append(weighted_loss.detach().item())

        epoch_loss = np.array(epoch_loss).mean()

        if epoch%10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {1+epoch}:")
            if do_global_reconstruction:
                print("train: last batch loss {:.2f} = {:.2f} patch + {:.2f} global".format(weighted_loss, patch_loss, global_loss))
                print("mu {:.2f}%".format(mu*100), "lr ratio {:.2f}%".format(100*current_lr/base_lr))
            else:
                print("train: last batch loss {:.2f}".format(weighted_loss))
                print("lr ratio {:.2f}%".format(100*current_lr/base_lr))
        
            #train visualisation
            plot_images_from_batch(image, mask, result["patchwise_reconstructions"], 1, config, save_to=f"{run_name}/{epoch}_train_patch.png")
            if do_global_reconstruction:
                plot_global_comparison(image[0], result["cls_reconstructed_image"][0], save_to=f"{run_name}/{epoch}_train_image.png")

            #valid visualisation
            model.eval()
            valid_batch = next(iter(valid_dl))
            image = valid_batch['image'][:, None, :, :]
            mask = (torch.zeros((image.shape[0], num_patches)).float().uniform_(0, 1) < mask_ratio).int()
            mask = mask.to(device)
            
            result = model(image, bool_masked_pos=mask)
            plot_images_from_batch(image, mask, result["patchwise_reconstructions"], 1, config, save_to=f"{run_name}/{epoch}_valid_patch.png")
            if do_global_reconstruction:
                plot_global_comparison(image[0], result["cls_reconstructed_image"][0], save_to=f"{run_name}/{epoch}_valid_image.png")
            model.train()

    import datetime
    torch.save(model.state_dict(), str(datetime.datetime.now()) + ".pth")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()