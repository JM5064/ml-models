"""Usage:
python -m models.BlazePoseFreiHAND.train_main
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
from .train import train, to_device, load_checkpoint
from datasets.FreiHAND.freihand_dataset import FreiHAND
# from datasets.MPII.random_affine import RandomAffine
# from datasets.MPII.random_horizontal_flip import RandomHorizontalFlip
# from datasets.MPII.random_occlusion import RandomOcclusion

from .losses.combined_loss import CombinedLoss


if __name__ == "__main__":
    random.seed(5064)
    torch.manual_seed(5064)
    np.random.seed(5064)

    train_transform = v2.Compose([
        # RandomHorizontalFlip(0.5, seed=5064),
        # RandomAffine(degrees=25, translate=None, scale=(0.75, 1.25), shear=0.1, seed=5064),
        # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # RandomOcclusion(0.1, 0.3, 0.5, seed=5064),
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    # train_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training/rgb'
    # val_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'

    # train_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training_xyz.json'
    # train_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training_K.json'
    # train_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/training_scale.json'

    # val_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    # val_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    # val_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'

    # 64 images for testing
    train_images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/rgb'
    val_images_dir = train_images_dir

    train_kpts_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_xyz.json'
    train_intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_K.json'
    train_scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_scale.json'

    val_kpts_json = train_kpts_json
    val_intrinsics_json = train_intrinsics_json
    val_scale_json = train_scale_json


    train_dataset = FreiHAND(
        images_dir=train_images_dir, 
        keypoints_json=train_kpts_json, 
        intrinsics_json=train_intrinsics_json,
        scale_json=train_scale_json,
        transform=train_transform)
    
    val_dataset = FreiHAND(
        images_dir=val_images_dir, 
        keypoints_json=val_kpts_json, 
        intrinsics_json=val_intrinsics_json,
        scale_json=train_scale_json,
        transform=train_transform)

    test_dataset = val_dataset

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False ,num_workers=1)

    num_keypoints = 21

    model = BlazePose(num_keypoints=num_keypoints)
    
    # Load in pretraining weights
    # weights = torch.load("models/BlazePose/runs_pretraining/30epochs/best.pt")
    # model.bb1.load_state_dict(weights["bb1"])
    # model.bb2.load_state_dict(weights["bb2"])

    # for param in model.bb1.parameters():
    #     param.requires_grad = False

    # for param in model.bb2.parameters():
    #     param.requires_grad = False

    model = to_device(model)

    adamW_params = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = optim.AdamW(model.parameters(), **adamW_params)

    def convnext_scheduler(optimizer, num_warmup_epochs, total_epochs):
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-num_warmup_epochs, eta_min=1e-5)

        if num_warmup_epochs == 0:
            return cosine_scheduler

        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_warmup_epochs, total_iters=num_warmup_epochs)

        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_epochs])

    warmup_epochs = 10
    total_epochs = 50
    scheduler = convnext_scheduler(optimizer, warmup_epochs, total_epochs)

    model, optimizer, scheduler, epoch = load_checkpoint("models/BlazePoseFreiHAND/runs/test/last.pt", model, optimizer, scheduler)

    train(
        model, 
        total_epochs, 
        start_epoch=epoch,
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        loss_func=CombinedLoss(), 
        optimizer=optimizer, 
        scheduler=scheduler
    )
