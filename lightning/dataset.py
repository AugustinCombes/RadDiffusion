import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import config
from torch import Tensor
from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
from torchvision.transforms import Compose, RandomResizedCrop, Normalize, ToTensor
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileLister,
    Shuffler,
    Batcher,
    Collator,
    Mapper,
    
)


IMG_TRANSFORM = nn.Sequential(
    RandomResizedCrop(scale=(0.9, 1.0), size=config.IMG_SHAPE, antialias=True),
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
        self.rs = MultiProcessingReadingService(num_workers=num_workers)

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_dp = self._get_base_datapipe("train")
        self.val_dp = self._get_base_datapipe("val")
        self.test_dp = self._get_base_datapipe("test")

    def _get_base_datapipe(self, split):
        split_dirs = self._get_split_dirs(split)
        files = FileLister(split_dirs, masks="*.h5", abspath=True)
        if split == "train": files = files.shuffle()
        images = CXRImageLoader(files)
        images = images.batch(self.batch_size, wrapper_class=np.array)
        images = images.collate(collate_fn=self._collate_fn)
        # images = images.map(fn=self._transform_fn)
        # images = images.batch(self.batch_size, wrapper_class=torch.stack)
        return images
    
    @staticmethod
    def _transform_fn(img) -> Tensor:
        return IMG_TRANSFORM(torch.from_numpy(np.array(img)).unsqueeze(0))
    
    @staticmethod
    def _collate_fn(img_batch: np.ndarray) -> Tensor:
        return IMG_TRANSFORM(torch.from_numpy(img_batch).unsqueeze(1))
        
    def _get_split_dirs(self, split):
        split_subdirs = {
            "train": ["p1%1i" % idx for idx in [0, 1, 2, 3, 4, 5, 6, 7]],
            "val": ["p18"],
            "test": ["p19"],
        }[split]
        return [os.path.join(self.data_dir, subdir) for subdir in split_subdirs]

    def _get_dataloader(self, dp):
        return DataLoader2(datapipe=dp, reading_service=self.rs)
    
    def train_dataloader(self):
        return self._get_dataloader(self.train_dp)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dp)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dp)
    

@functional_datapipe("load_cxr")
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
                for img_id in img_ids:
                    img = h5_file[img_id]
                    
                    # Yield image
                    if not self.load_text:
                        yield img
                    else:
                        txt = h5_file[study_id + "_txt"]
                        yield {"img": img, "txt": txt}
                        

if __name__ == "__main__":
    data_module = CXRDataModule(
        config.DATA_MAIN_DIR, config.BATCH_SIZE, config.NUM_WORKERS
    )
    data_pipe = data_module._get_base_datapipe()
    for image in data_pipe:
        print(image.shape)
