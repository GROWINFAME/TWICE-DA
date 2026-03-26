import albumentations as A
import torch
import torchvision
from core_twice.augmentations import AlbumentationsRandAugment, CutMixTransform, MixUpTransform
from torchvision import datasets
import torch.utils.data as data
import lightning.pytorch as pl
import cv2
import numpy as np
from torch.utils.data import DataLoader, default_collate
import os

class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def grayscale_to_rgb(self, image):
        if len(image.shape) == 2:
            return np.stack((image,) * 3, axis=-1)
        else:
            return image

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=self.grayscale_to_rgb(np.array(img)))['image']

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset,
                 dataset_path,
                 image_size,
                 batch_size,
                 num_classes):
        super().__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.train_transforms = A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LANCZOS4, p=1.0),
            A.HorizontalFlip(p=0.75),
            AlbumentationsRandAugment(N_TFMS=2, p=1.0),
            A.Normalize(),
            A.ToTensorV2()
        ])

        self.test_val_transforms = A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LANCZOS4, p=1.0),
            A.Normalize(),
            A.ToTensorV2()
        ])

    def setup(self, stage=None):
        self.reg_augment_transform = [
            MixUpTransform(num_classes=self.num_classes, p_mixup=0.50, alpha=1.0),
            CutMixTransform(num_classes=self.num_classes, p_cutmix=0.50, alpha=1.0)
        ]

        if self.dataset == 'cifar-100':
            self._setup_cifar100()
        elif self.dataset == 'caltech-256':
            self._setup_caltech256()
        elif self.dataset == 'imagenet':
            self._setup_imagenet()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _setup_cifar100(self):
        self.train_data = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True,
                                                        transform=Transforms(self.train_transforms),
                                                        download=True)
        self.validation_data = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True,
                                                             transform=Transforms(self.test_val_transforms),
                                                             download=False)
        self.test_data = torchvision.datasets.CIFAR100(root=self.dataset_path, train=False,
                                                       transform=Transforms(self.test_val_transforms),
                                                       download=True)
        if os.path.exists(os.path.join(self.dataset_path, 'indices.pth')):
            indices = torch.load(os.path.join(self.dataset_path, 'indices.pth'))
        else:
            indices = torch.randperm(len(self.train_data))
            torch.save(indices, os.path.join(self.dataset_path, 'indices.pth'))
        train_set_size = int(len(self.train_data) * 0.9)
        valid_set_size = len(self.train_data) - train_set_size
        self.train_data = torch.utils.data.Subset(self.train_data, indices[:-valid_set_size])
        self.validation_data = torch.utils.data.Subset(self.validation_data, indices[-valid_set_size:])

    def _setup_caltech256(self):
        self.train_data = torchvision.datasets.Caltech256(root=self.dataset_path,
                                                          transform=Transforms(self.train_transforms))
        self.validation_data = torchvision.datasets.Caltech256(root=self.dataset_path,
                                                               transform=Transforms(self.test_val_transforms))
        self.test_data = torchvision.datasets.Caltech256(root=self.dataset_path,
                                                         transform=Transforms(self.test_val_transforms))
        if os.path.exists(os.path.join(self.dataset_path, 'indices.pth')):
            indices = torch.load(os.path.join(self.dataset_path, 'indices.pth'))
        else:
            indices = torch.randperm(len(self.train_data))
            torch.save(indices, os.path.join(self.dataset_path, 'indices.pth'))
        total_len = len(self.train_data)
        train_len = int(0.8 * total_len)
        val_len = int(0.10 * total_len)
        test_len = int(0.10 * total_len)
        self.train_data = torch.utils.data.Subset(self.train_data, indices[:train_len])
        self.validation_data = torch.utils.data.Subset(self.validation_data, indices[train_len:train_len + val_len])
        self.test_data = torch.utils.data.Subset(self.test_data, indices[train_len + val_len:train_len + val_len + test_len])

    def _setup_imagenet(self):
        self.train_data = torchvision.datasets.ImageNet(root=self.dataset_path, split='train',
                                                        transform=Transforms(self.train_transforms))
        self.validation_data = torchvision.datasets.ImageNet(root=self.dataset_path, split='val',
                                                             transform=Transforms(self.test_val_transforms))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def collate_fn(self, batch):
        reg_augment = np.random.choice(self.reg_augment_transform, 1)[0]
        return reg_augment.transform(*default_collate(batch))