import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import torchvision
from datasets.CIFAR10.cifar10 import CIFAR10

from models.MetaQNN.train import train
from models.MetaQNN.metaqnn import MetaQNN
from models.MetaQNN.config.rl_config import *
from models.utils import DEVICE


def convnext_scheduler(optimizer, num_warmup_epochs, total_epochs):
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_warmup_epochs, total_iters=num_warmup_epochs)

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-num_warmup_epochs, eta_min=1e-5)

    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_epochs])


def main():
    # Define transformations
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=0, translate=(0.156, 0.156)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    # Fetch datasets and create train/val/test split
    cifar10_train = torchvision.datasets.CIFAR10(root='datasets/CIFAR10/CIFAR10', train=True, download=True, transform=None)
    cifar10_test = torchvision.datasets.CIFAR10(root='datasets/CIFAR10/CIFAR10', train=False, download=True, transform=transform)

    indices = list(range(len(cifar10_train)))
    val_size = int(len(cifar10_train) * 0.1)

    train_indices, val_indices = train_test_split(indices, test_size=val_size, shuffle=True, random_state=5064)

    train_set = CIFAR10(images=cifar10_train.data[train_indices], labels=[cifar10_train.targets[i] for i in train_indices], transform=train_transform)
    val_set = CIFAR10(images=cifar10_train.data[val_indices], labels=[cifar10_train.targets[i] for i in val_indices], transform=transform)

    # Create dataloaders
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_set, batch_size=128, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset=cifar10_test, batch_size=128, shuffle=False, num_workers=1)


    layer_configs = [
        {'layer_type': 0, 'out_channels': 256, 'kernel_size': 3, 'layer_depth': 1, 'representation_size': 32},
        {'layer_type': 0, 'out_channels': 256, 'kernel_size': 5, 'layer_depth': 2, 'representation_size': 32},
        {'layer_type': 1, 'kernel_size': 2, 'stride': 2, 'layer_depth': 3, 'representation_size': 16},
        {'layer_type': 0, 'out_channels': 256, 'kernel_size': 3, 'layer_depth': 4, 'representation_size': 16},
        {'layer_type': 0, 'out_channels': 256, 'kernel_size': 5, 'layer_depth': 5, 'representation_size': 16},
        {'layer_type': 1, 'kernel_size': 3, 'stride': 2, 'layer_depth': 6, 'representation_size': 7},
        {'layer_type': 0, 'out_channels': 64, 'kernel_size': 1, 'layer_depth': 7, 'representation_size': 7},
        {'layer_type': 3}
    ]

    model = MetaQNN(layer_configs=layer_configs, input_size=32, input_channels=3)
    model = model.to(DEVICE)

    adamW_params = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = optim.AdamW(model.parameters(), **adamW_params)
    
    total_epochs = 50
    scheduler = convnext_scheduler(optimizer, 5, total_epochs)

    def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        return model, optimizer, scheduler, checkpoint['epoch']
    
    # model, optimizer, scheduler, epoch = load_checkpoint("models/MetaQNN/logs/0.8098/last.pt", model, optimizer, scheduler)


    train(model=model, num_epochs=total_epochs, train_loader=train_loader, val_loader=val_loader, 
          loss_func=nn.CrossEntropyLoss(), optimizer=optimizer, scheduler=scheduler, 
          test_loader=test_loader, val_on=True, save_folder='models/MetaQNN/logs/0.8098/0.8098')
    

if __name__ == "__main__":
    main()