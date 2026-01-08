import torch
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# added this to prevent some crashing
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except Exception as e:
            path, _ = self.samples[index]
            print(f"\n[X] Skipping corrupted/invalid file: {path}")
            # try to return the next image instead of crashing
            return self.__getitem__((index + 1) % len(self))

def get_loaders(split_root, batch_size=32):
    # DINOv3, ResNet-50, and ViT-Base all use ImageNet normalization and expect 224x224 input
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # creating standard ImageFolder datasets from the stratified physical split
    image_datasets = {
        'train': SafeImageFolder(f"{split_root}/train", data_transforms['train']),
        'val': SafeImageFolder(f"{split_root}/val", data_transforms['val_test']),
        'test': SafeImageFolder(f"{split_root}/test", data_transforms['val_test'])
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6)
                   for x in ['train', 'val', 'test']}


    return dataloaders