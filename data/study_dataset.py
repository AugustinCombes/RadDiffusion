import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

import pandas as pd
import numpy as np

import random
from glob import glob
import h5py

from pytorch_lightning import LightningDataModule

class DicomStudyDataset(IterableDataset):
    def __init__(
            self, 
            mimic_cxr_path, 
            mode="train", 
            device="cuda",
            include_text=False,
            include_labels=False,
            ):

        assert mode in ["train", "valid", "test"]

        self.device = device
        
        # split at patient level: train=p10->7, valid=p18, test=p19
        if mode == "train":
            self.h5_files = glob("/".join([mimic_cxr_path, "p1[0-7]", "*.h5"]))
            # self.h5_files = glob("/".join([mimic_cxr_path, "p10", "*.h5"]))
        elif mode == "valid":
            self.h5_files = glob("/".join([mimic_cxr_path, "p18", "*.h5"]))
        else: #mode == "test"
            self.h5_files = glob("/".join([mimic_cxr_path, "p19", "*.h5"]))

        self.include_text = include_text
        self.include_labels = include_labels
        self.labels = [
            'Atelectasis', 
            'Cardiomegaly', 
            'Consolidation', 
            'Edema',
            'Enlarged Cardiomediastinum', 
            'Fracture', 
            'Lung Lesion', 
            'Lung Opacity',
            'No Finding', 
            'Pleural Effusion', 
            'Pleural Other', 
            'Pneumonia',
            'Pneumothorax',
            'Support Devices'
            ]

        #labels
        df = pd.read_csv("../scrap/mimic-cxr-2.0.0-chexpert.csv")
        df = df.set_index('study_id')
        df = df.drop(columns=["subject_id"])

        df = df[self.labels]

        ##trans1
        # df = df.where(df != -1.0, -1) 
        # df = df.where(~df.isna(), -1) 

        ##trans2 #Maps 1->1 and rest->0
        # df = df.where(df == 1.0, 0) 
        # self.label_means = [0.1996, 0.1996, 0.0460, 0.1163, 0.0330, 0.0193, 0.0302, 0.2278, 0.3333, 0.2390, 0.0068, 0.0710, 0.0503, 0.2933]

        ##trans3 #Maps 1 and -1 -> 1 and nan, 0 -> 0
        df = df.where(df != -1.0, 1.0) 
        df = df.where(df == 1.0, 0)
        self.label_means = [0.2456, 0.2232, 0.0668, 0.1770, 0.0728, 0.0219, 0.0325, 0.2419, 0.3312, 0.2623, 0.0125, 0.1529, 0.0501, 0.2932]

        df = df.astype(int)
        
        self.study2labels = {x[0]: list(x)[1:] for x in df.to_records()}

        # if mode == "train":
        #     self.transform = transforms.Compose([
        #         transforms.RandomCrop(256),
        #         transforms.RandomRotation(10)
        #     ])
        # else:
        #     self.transform = transforms.Compose([])
    
    def __len__(self):
        length = 0
        self.h5_iterator = iter(self.h5_files)
        
        for hdf5_path in self.h5_iterator:
            try:
                hdf5_file = h5py.File(hdf5_path, "r")
            except OSError as e:
                continue
            keys = list(hdf5_file.keys())
            length += len(set(map(lambda x: x.split("_")[0], keys)))
        
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

        for hdf5_path in self.h5_iterator:
            try:
                hdf5_file = h5py.File(hdf5_path, "r")
            except OSError as e:
                continue

            keys = list(hdf5_file.keys())
            unique_studies = list(set(map(lambda x: x.split("_")[0], keys)))
            for study_id in unique_studies:
                if self.include_text:
                    text_id = "_".join([study_id, "txt"])
                    text = torch.tensor(hdf5_file[text_id][...], device=self.device)

                if self.include_labels: # multi expert label
                    labels = self.study2labels.get(int(study_id[1:]), False)
                    if not labels:
                        continue

                image_ids = filter(lambda x: study_id in x and "txt" not in x, keys)
                # image = torch.stack(list(map(
                #     lambda x: self.transform(
                #         torch.tensor(
                #             hdf5_file[x][...],
                #             device=self.device
                #         ).unsqueeze(0)
                #     ),
                #     image_ids
                #     ))) # [n_image, channels, height, width]
                image = torch.stack(list(map(
                    lambda x: torch.tensor(
                        hdf5_file[x][...],
                        device=self.device
                        ).unsqueeze(0)
                    , image_ids
                    ))) # [n_image, channels, height, width]
                sample = {"image": image}
                if self.include_text:
                    sample["text"] = text
                if self.include_labels:
                    sample["labels"] = torch.tensor(labels, device=self.device)
                yield sample

    def get_by_pid(self, pid, device):
        pid_h5_file = list(filter(lambda x: str(pid) in x, self.h5_files))
        assert len(pid_h5_file) == 1, f"found {len(pid_h5_file)} files for pid p{pid}"

        studies = []
        hdf5_file = h5py.File(pid_h5_file[0], "r")
        keys = list(hdf5_file.keys())
        unique_studies = list(set(map(lambda x: x.split("_")[0], keys)))
        for study_id in unique_studies:
            if self.include_text:
                text_id = "_".join([study_id, "txt"])
                text = torch.tensor(hdf5_file[text_id][...], device=device)

            if self.include_labels: # multi expert label
                labels = self.study2labels.get(int(study_id[1:]), False)
                if not labels:
                    continue

            image_ids = filter(lambda x: study_id in x and "txt" not in x, keys)

            # image = [torch.tensor(hdf5_file[image_id][...], device=device).unsqueeze(0) for image_id in image_ids]
            image = torch.stack(list(map(
                    lambda x: torch.tensor(
                        hdf5_file[x][...],
                        device=device
                        ).unsqueeze(0),
                    image_ids
                    ))) # [n_image, channels, height, width]
            sample = {"image": image}
            if self.include_text:
                sample["text"] = text
            if self.include_labels:
                sample["labels"] = torch.tensor(labels, device=device)
            studies.append(sample)
        
        # images = [item['image'] for item in studies] #*
        # images = torch.stack(list(map(lambda x: x['image'], studies))) # [n_image, channels, height, width] # pê ancien
        images = torch.cat(list(map(lambda x: x['image'], studies)))
        lengths = torch.tensor(list(map(lambda x: len(x['image']), studies)), device=device)

        res = {
            "images": images,
            "lengths": lengths
            }
        if "text" in studies[0].keys():
            res["text"] = pad_sequence([item['text'] for item in studies], batch_first=True, padding_value=0)
        if "labels" in studies[0].keys():
            res["labels"] = torch.stack([item['labels'] for item in studies])
        return res

def collate_fn(batch):
    # images = [item['image'] for item in batch] #*
    
    images = torch.cat(list(map(lambda x: x['image'], batch))) # [variable, channels, height, width]
    lengths = torch.tensor(list(map(lambda x: len(x['image']), batch))) # [batch_size]

    res = {
        "images": images,
        "lengths": lengths,
        }
    
    if "text" in batch[0].keys():
        res["text"] = pad_sequence([item['text'] for item in batch], batch_first=True, padding_value=0)
    if "labels" in batch[0].keys():
        res["labels"] = torch.stack([item['labels'] for item in batch])
        
    return res

def get_splitted_dataloaders(
        mimic_cxr_path, 
        device, 
        batch_size, 
        num_workers=2,
        **kwargs
        ):
    
    train_dataset = DicomStudyDataset(mimic_cxr_path, mode="train", device=device, **kwargs)
    valid_dataset = DicomStudyDataset(mimic_cxr_path, mode="valid", device=device, **kwargs)
    test_dataset = DicomStudyDataset(mimic_cxr_path, mode="test", device=device, **kwargs)

    #num_workers' rule of thumb: num_worker = 4 * num_GPU
    train_dataloader = DataLoader(train_dataset, 
        batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, 
        batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 
        batch_size=batch_size, drop_last=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader