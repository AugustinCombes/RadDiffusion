import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoImageProcessor

import pandas as pd
import numpy as np

import random
from glob import glob
import h5py

class DicomDataset(IterableDataset):
    def __init__(self, mimic_cxr_path, mode="train", device="cuda"):

        assert mode in ["train", "valid", "test"]

        self.device = device
        cxr_regex = "/".join([mimic_cxr_path, "p*", "*.h5"])
        self.h5_files = glob(cxr_regex)
        self.h5_files.sort()

        n_sample = len(self.h5_files)
        valid_treshold, test_treshold = int(0.8*n_sample), int(0.9*n_sample)
        # later : train=p10->7, valid=p18, test=p19
        
        if mode == "train":
            self.h5_files = self.h5_files[:valid_treshold]
        elif mode == "valid":
            self.h5_files = self.h5_files[valid_treshold:test_treshold]
        else:
            self.h5_files = self.h5_files[test_treshold:]

        self.num_patches = 196

        #labels
        df = pd.read_csv("../scrap/mimic-cxr-2.0.0-chexpert.csv")

        df = df.where(df!=0, 1)
        df = df.where(df!=-1, np.nan)
        df = df.where(~df.isna(), 0)

        df = df.astype(int)
        df = df.drop(columns=["subject_id"])
        self.study2labels = {x[0]: list(x)[1:] for x in  df.set_index('study_id').to_records()}

    def prob_mask_like(self, inputs, mask_ratio=0.15):
        return torch.zeros_like(inputs).float().uniform_(0, 1) < mask_ratio
    
    def __len__(self):
        length = 0
        self.h5_iterator = iter(self.h5_files)
        
        for hdf5_path in self.h5_iterator:
            try:
                hdf5_file = h5py.File(hdf5_path, "r")
            except OSError as e:
                continue
            keys = list(hdf5_file.keys())
            unique_studies = list(set(map(lambda x: x.split("_")[0], keys)))
            for study_id in unique_studies:
                image_ids = filter(lambda x: study_id in x and "txt" not in x, keys)
                length += len(list(image_ids))
        
        return length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
            worker_total_num = 1
        else:  # in a worker process
            worker_id = worker_info.id
            worker_total_num = worker_info.num_workers

        self.h5_iterator = iter(self.h5_files[worker_id::worker_total_num])  # Split the files among the workers
        ##possible de shuffler ici-haut

        for hdf5_path in self.h5_iterator:
            try:
                hdf5_file = h5py.File(hdf5_path, "r")
            except OSError as e:
                continue

            keys = list(hdf5_file.keys())
            unique_studies = list(set(map(lambda x: x.split("_")[0], keys)))
            for study_id in unique_studies:
                text_id = "_".join([study_id, "txt"])
                text = torch.tensor(hdf5_file[text_id][...], device=self.device)

                # multi expert label
                target = self.study2labels.get(int(study_id[1:]), False)
                if not target:
                    continue

                image_ids = filter(lambda x: study_id in x and "txt" not in x, keys)
                for image_id in image_ids:
                    image = torch.tensor(hdf5_file[image_id][...], device=self.device)
                    sample = {
                        "image": image,
                        "text": text,
                        "target": torch.tensor(target, device=self.device),
                    }
                    yield sample

def collate_fn(batch):
    return {
        "text": pad_sequence([item['text'] for item in batch], batch_first=True, padding_value=0),
        "image": torch.stack([item['image'] for item in batch]),
        "target": torch.stack([item['target'] for item in batch])
    }

def get_splitted_dataloaders(mimic_cxr_path, device, batch_size, num_workers=2):
    train_dataset = DicomDataset(mimic_cxr_path, mode="train", device=device)
    valid_dataset = DicomDataset(mimic_cxr_path, mode="valid", device=device)
    test_dataset = DicomDataset(mimic_cxr_path, mode="test", device=device)

    #num_workers' rule of thumb: num_worker = 4 * num_GPU
    if num_workers > 0:
        train_dataloader = DataLoader(train_dataset, 
            batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True)
        valid_dataloader = DataLoader(valid_dataset, 
            batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True)
    else:
        train_dataloader = DataLoader(train_dataset, 
            batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, 
            batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, collate_fn=collate_fn)
    return train_dataloader, valid_dataloader, test_dataloader