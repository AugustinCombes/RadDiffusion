import torch.multiprocessing as mp

from models.MAE_GAN import MAE_GAN
from dicom_dataset import get_splitted_dataloaders
from utils import plot_images_from_batch, plot_global_comparison
from configs import configs

import math
from tqdm import tqdm
from itertools import chain
import logging
import argparse

import torch
from torch import nn
import numpy as np
import transformers
from transformers import get_cosine_schedule_with_warmup

import os
import datetime
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default=None, help='Name of the run, if any.')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
parser.add_argument('--device', type=str, default="cuda:2", help='Gpu to use.')
parser.add_argument('--model', type=str, default="xs", help='Size of the ViT backbone.')
args = parser.parse_args()

run_name = os.path.join("runs", args.run_name)

while os.path.exists(run_name) and run_name is not None:
    run_name += "_"

def main():
    device = args.device
    
    print(f'Run Name: {run_name or "none"}')
    print(f"Number of Workers: {args.num_workers}")
    
    if run_name is not None:
        os.mkdir(run_name), [os.mkdir(os.path.join(run_name, f"vis_{idx}")) for idx in range(5)]
        logging.basicConfig(filename=f'{run_name}/logs.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    config = configs[args.model]

    model = MAE_GAN(config).to(device)
    print(f"Number of million parameters: {model.num_parameters()/1e6}")
    model.train()

    batchSize = 1024
    epochs = 300 #30 #60 #501
    warmup_epochs = 50
    base_lr = 1e-4 #5e-4
    weight_decay = 1e-5 #5e-2
    
    train_dl, valid_dl, _ = get_splitted_dataloaders("preprocessed_data/xcr_256", device, batchSize, num_workers=args.num_workers)

    g_optimizer = torch.optim.Adam(chain(model.vit.parameters(), model.decoder.parameters()), lr=base_lr, weight_decay=weight_decay)
    bce_log = torch.nn.BCEWithLogitsLoss()

    n_sample = 180776 #len(train_dl.dataset) #speed-up pour 256

    n_steps_per_epoch = math.ceil(n_sample/batchSize)
    warmup_steps = n_steps_per_epoch * warmup_epochs
    training_steps = n_steps_per_epoch * (epochs - warmup_epochs)

    n_steps = n_steps_per_epoch * epochs
    pbar = tqdm(total=n_steps, desc="Batch steps")

    # Define reference image & noise for visualisation 
    batch_ref = next(iter(train_dl))
    seq_length = (model.config.image_size ** 2) // (model.config.patch_size ** 2)
    noise_ref = torch.rand(batchSize, seq_length, device=model.device)

    g_scheduler = get_cosine_schedule_with_warmup(g_optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps, num_cycles=1.5)
    step = 0
    for epoch in range(epochs):
        pbar.set_postfix(Epoch=epoch+1, refresh=True)
        epoch_stats = {
            "L_2": [],
        }

        for idx, batch in enumerate(train_dl):
            step +=1
            
            #Generator
            g_optimizer.zero_grad()
            result = model(batch)
            
            l2_loss = result["loss"]
            l2_loss.backward()
            
            g_optimizer.step(), g_scheduler.step()

            if True:
                epoch_stats["L_2"].append(l2_loss.detach().item())
                
            pbar.update(1)
        epoch_stats = {k: np.array(v).mean() for k,v in epoch_stats.items()}

        if run_name is not None:
            logging.info('')
            logging.info(f'Epoch {epoch}')
            
            row_l2 = 'l_mae -> {:.2f}'.format(epoch_stats["L_2"])
            logging.info(row_l2)
        
            with torch.no_grad():
                res = model(batch_ref, noise=noise_ref)
                for ex in range(5):
                    model.save_img(
                        model.get_nth_boxed_visualisation(res, ex), 
                        f"{run_name}/vis_{ex}/epoch_{epoch}.jpg"
                    )

    if not run_name is not None:
        torch.save(model.state_dict(), str(datetime.datetime.now()) + ".pth")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()