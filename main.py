import argparse

from data.study_dataset import get_splitted_dataloaders
from src.pretrain import pretrain
from src.supervised import supervised

import torch
import torch.multiprocessing as mp
import os
pjoin = os.path.join
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--ref_name', type=str, default='untitled', help='Name of the run.')
parser.add_argument('--scheme', type=str, default="vanilla", help='Scheme used to pretrain the model.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers.')
parser.add_argument('--device', type=list, default=[1, 2], help='Gpu devices to use.')
parser.add_argument('--encoder_size', type=str, default="xs", help='Size of the ViT backbone.')
args = parser.parse_args()

class PretrainCFG:
    name = args.ref_name
    pretraining_scheme = args.scheme
    decoder_type = 'cross-self'
    config = args.encoder_size
    batchSize = 512
    epochs = 200
    base_lr = 1.4e-4
    weight_decay = 0.05
    
class FinetuneCFG:
    name = args.ref_name
    config = args.encoder_size
    batchSize = 512
    epochs = 20
    base_lr = 5e-5
    weight_decay = 0.05
    patience = 5

def main():
    torch.manual_seed(0)

    dataloaders = get_splitted_dataloaders(
        "preprocessed_data/xcr_256",
        "cpu",
        PretrainCFG.batchSize,
        num_workers=args.num_workers,
        include_labels=True
        )
    checkpoint_path = pjoin("checkpoints", args.ref_name)
    

    #*# PRETRAINING #*#
    if not os.path.exists(checkpoint_path) or args.ref_name == 'untitled':
        pretrain(PretrainCFG, args, dataloaders)
    else:
        print(f'Found reference pretrain encoder weights at path {checkpoint_path}. Running only supervised inference.')


    #*# FINE-TUNING #*#
    supervised(FinetuneCFG, args, dataloaders)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()