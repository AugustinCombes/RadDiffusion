import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import config
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop, Normalize
from torchdata.datapipes.iter import IterDataPipe, FileLister, Shuffler


IMG_TRANSFORM = nn.Sequential(
    RandomResizedCrop(scale=(0.9, 1.0), size=(256, 256), antialias=True),
    Normalize(
        [0.0 for _ in range(config.IMG_CHANNELS)],
        [1.0 for _ in range(config.IMG_CHANNELS)],
    ),
)


class CXRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_ds = self._get_base_datapipe("train")
        self.val_ds = self._get_base_datapipe("val")
        self.test_ds = self._get_base_datapipe("test")

    def _get_base_datapipe(self, split):
        split_dirs = self._get_split_dirs(split)
        files = FileLister(split_dirs, masks="*.h5", abspath=True)
        # files = Shuffler(files)
        return CXRImageLoader(files)

    def _get_split_dirs(self, split):
        split_subdirs = {
            "train": ["p1%1i" % idx for idx in [0, 1, 2, 3, 4, 5, 6, 7]],
            "val": ["p18"],
            "test": ["p19"],
        }[split]
        return [os.path.join(self.data_dir, subdir) for subdir in split_subdirs]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class CXRImageLoader(IterDataPipe):
    def __init__(self, dp_files, load_text=False):
        super().__init__()
        self.dp_files = dp_files
        self.load_text = load_text
        
    def __iter__(self):
        for h5_path in self.dp_files:
            # Find studies in h5 file
            h5_file = h5py.File(h5_path, "r")
            h5_keys = h5_file.keys()
            unique_studies = set(map(lambda x: x.split("_")[0], h5_keys))
            for study_id in unique_studies:
                # Load image(s)
                img_ids = list(
                    filter(lambda x: study_id in x and "txt" not in x, h5_keys)
                )
                img = h5_file[img_ids[0]]  # first image only, for now
                
                # Yield data (adding associated text if required)
                if not self.load_text:
                    yield self._process_img_data(img)
                else:
                    txt = h5_file[study_id + "_txt"]
                    yield {
                        "img": self._process_img_data(img),
                        "txt": self._process_txt_data(txt),
                    }

    @staticmethod
    def _process_img_data(h5_data):
        torch_img = torch.from_numpy(np.array(h5_data)).unsqueeze(0)
        return IMG_TRANSFORM(torch_img)

    @staticmethod
    def _process_txt_data(h5_data):
        return torch.from_numpy(np.array(h5_data))


if __name__ == "__main__":
    data_module = CXRDataModule(
        config.DATA_MAIN_DIR, config.BATCH_SIZE, config.NUM_WORKERS
    )
    data_pipe = data_module._get_base_datapipe()
    for image in data_pipe:
        print(image.shape)
