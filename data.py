from typing import Tuple

import pytorch_lightning as pl

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torch.utils.data import DataLoader, random_split

from dataloader.dataset import DeepClusteringDataset

class DeepClusteringDataModule(pl.LightningDataModule):
    def __init__(self,
                data_dir: str,
                batch_size: int = 8,
                num_workers: int = 4,
                image_size: Tuple[int, int] = (224, 224)):
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.__split_dataset()
        
    @property
    def dataset_transforms(self):
        return transforms.Compose([
                    transforms.Resize(self.image_size, interpolation = InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])


    def __split_dataset(self):
        dataset = DeepClusteringDataset(data_dir=self.data_dir,
                                        transform=self.dataset_transforms)

        train_sample = int(len(dataset)*0.8)
        val_sample = len(dataset) - train_sample
        self.train_set, self.val_set = random_split(dataset, [train_sample, val_sample])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

if __name__ == '__main__':
    import cv2
    
    data_dir = '/home/santapo/OnlineLab/deep_clustering/data/no_label/images_mr'
    dm = DeepClusteringDataModule(data_dir=data_dir)
    sample = dm.train_dataloader().dataset[1].numpy().transpose(1,2,0)
    cv2.imshow('', sample)
    cv2.waitKey(0)