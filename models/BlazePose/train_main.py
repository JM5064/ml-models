"""Usage:
python -m models.BlazePose.train_main
"""

import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from .blazepose import BlazePose
from .train import train, to_device
from datasets.MPII.mpii_dataset import MPIIDataset
from datasets.MPII.random_affine import RandomAffine
from datasets.MPII.random_horizontal_flip import RandomHorizontalFlip
from datasets.MPII.random_occlusion import RandomOcclusion

from .losses.combined_loss import CombinedLoss


if __name__ == "__main__":
    random.seed(5064)
    torch.manual_seed(5064)
    np.random.seed(5064)

    train_transform = v2.Compose([
        RandomHorizontalFlip(0.5, seed=5064),
        RandomAffine(degrees=25, translate=(0.15, 0.15), scale=(0.75, 1.25), shear=0.1, seed=5064),
        # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        RandomOcclusion(0.1, 0.3, 0.5, seed=5064),
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    images_dir = 'datasets/MPII/mpii/images'

    train_json = 'datasets/MPII/mpii/med_train.json'
    val_json = 'datasets/MPII/mpii/med_val.json'
    test_json = 'datasets/MPII/mpii/mini_val.json'

    train_dataset = MPIIDataset(images_dir, train_json, transform=train_transform)
    val_dataset = MPIIDataset(images_dir, val_json, transform=transform)
    test_dataset = MPIIDataset(images_dir, test_json, transform=transform)

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False ,num_workers=1)

    num_keypoints = 16

    model = BlazePose(num_keypoints=num_keypoints)
    model = to_device(model)

    adamW_params = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = optim.AdamW(model.parameters(), **adamW_params)

    def convnext_scheduler(optimizer, num_warmup_epochs, total_epochs):
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_warmup_epochs, total_iters=num_warmup_epochs)

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-num_warmup_epochs, eta_min=1e-5)

        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_epochs])

    warmup_epochs = 2
    total_epochs = 6
    scheduler = convnext_scheduler(optimizer, warmup_epochs, total_epochs)

    train(
        model, 
        total_epochs, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        loss_func=CombinedLoss(), 
        optimizer=optimizer, 
        scheduler=scheduler
    )
