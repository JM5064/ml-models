"""Usage:
python -m models.BlazePose.pretrain_main
"""

import random
import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from .blazepose_pretraining import BlazePose

def to_device(obj):
    if torch.cuda.is_available():
        obj = obj.to("cuda")
    elif torch.backends.mps.is_available():
        obj = obj.to("mps")

    return obj


def log_results(file, metrics):
    for metric in metrics:
        file.write(f'{metric}: {metrics[metric]}\t')

    file.write('\n')


def compute_class_weights(train_dir_path):
    counts = {}
    classes = []
    total = 0

    for pose in os.listdir(train_dir_path):
        if not os.path.isdir(f'{train_dir_path}/{pose}'):
            continue

        counts[pose] = len(os.listdir(f'{train_dir_path}/{pose}'))
        classes.append(pose)
        total += counts[pose]

    classes.sort()
    weights = [total / (len(classes) * counts[pose]) for pose in classes]

    return torch.tensor(weights, dtype=torch.float)


def validate(model, val_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = to_device(inputs)
            labels = to_device(labels)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()

            _, predictions = torch.max(outputs, 1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_val_loss = running_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    true_labels = set(np.unique(all_labels))
    predicted_labels = set(np.unique(all_preds))

    missing_labels = true_labels - predicted_labels
    print("Missing labels:", missing_labels)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_val_loss": average_val_loss,
        "confusion_matrix": conf_matrix
    }

    return metrics


def train(
        model,
        num_epochs,
        train_loader,
        val_loader,
        test_loader,
        loss_func,
        optimizer,
        scheduler,
        start_epoch=0,
        runs_dir="models/BlazePose/runs_pretraining",
        augmentations=None
    ):
    # create log file
    time = str(datetime.now())
    os.mkdir(runs_dir + "/" + time)
    logfile = open(runs_dir + "/" + time + "/metrics.txt", "a")
    best_val_loss = float('inf')

    # training loop
    for i in range(start_epoch, num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            # Apply batch augmentations like cutmix or mixup
            if augmentations:
                for augmentation in augmentations:
                    inputs, labels = augmentation(inputs, labels)

            inputs = to_device(inputs)
            labels = to_device(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print and log metrics
        average_train_loss = running_loss / len(train_loader)
        metrics = validate(model, val_loader, loss_func)
        metrics["average_train_loss"] = average_train_loss
        del metrics["confusion_matrix"]


        print(f'Epoch {i+1} Results:')
        print(f'Train Loss: {average_train_loss}\tValidation Loss: {metrics["average_val_loss"]}')
        print(f'Accuracy: {metrics["accuracy"]}\tPrecision: {metrics["precision"]}\tRecall: {metrics["recall"]}\tF1-score: {metrics["f1"]}')

        log_results(logfile, metrics)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # save best model
        checkpoint = {
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            "bb1": model.bb1.state_dict(), # save the first two layers
            "bb2": model.bb2.state_dict(),
            "bb3": model.bb3.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        average_val_loss = metrics["average_val_loss"]
        if average_val_loss < best_val_loss:
            torch.save(checkpoint, runs_dir + "/" + time + "/best.pt")
            best_val_loss = average_val_loss

        torch.save(checkpoint, runs_dir + "/" + time + "/last.pt")

    # test model and print/log testing metrics
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'Accuracy: {metrics["accuracy"]}\tPrecision: {metrics["precision"]}\tRecall: {metrics["recall"]}\tF1-score: {metrics["f1"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')

    test_logfile = open(runs_dir + "/" + time + "/test_metrics.txt", "a")
    log_results(test_logfile, metrics)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch']


if __name__ == "__main__":
    random.seed(5064)

    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.Resize((256, 256)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(),
        v2.RandAugment(),
    ])

    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    dataset_path = 'datasets/Caltech-256/Caltech-256/'

    train_dataset = datasets.ImageFolder(dataset_path + 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_path + 'val', transform=transform)
    test_dataset = datasets.ImageFolder(dataset_path + 'test', transform=transform)

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False ,num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False ,num_workers=1)

    num_classes = 257

    model = BlazePose(num_classes=num_classes)
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

    warmup_epochs = 5
    total_epochs = 30
    scheduler = convnext_scheduler(optimizer, warmup_epochs, total_epochs)

    class_weights = to_device(compute_class_weights(dataset_path + 'train'))

    cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
    mixup = v2.MixUp(alpha=0.8, num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    augmentations = [cutmix_or_mixup]

    train(model, total_epochs, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, loss_func=nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1), optimizer=optimizer, scheduler=scheduler, augmentations=augmentations)
