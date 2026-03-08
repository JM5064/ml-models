import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
from metrics.mpjpe import mpjpe_3D
from metrics.pck import pck_2D, pck_3D


def to_device(obj):
    if torch.cuda.is_available():
        obj = obj.to("cuda")
    elif torch.backends.mps.is_available():
        obj = obj.to("mps")

    return obj


def log_results(file_path, metrics):
    # Open file
    file = open(file_path, "a")

    # Create header if file is blank
    if os.path.getsize(file_path) == 0:
        for metric in metrics:
            file.write(f'{metric},')
        
        file.write('\n')

    # Log metrics
    for metric in metrics:
            file.write(f'{metrics[metric]},')

    file.write('\n')
    # Makes file update immediately
    file.flush()
    os.fsync(file.fileno())


def validate(model, val_loader, loss_func, image_size):
    model.eval()
    all_preds = []
    all_labels = []
    total_combined_loss = 0.0
    total_regression_loss = 0.0
    total_heatmap_loss = 0.0
    total_offset_loss = 0.0

    mpjpe = 0.0
    pck20 = 0.0
    pck40 = 0.0

    with torch.no_grad():
        for inputs, keypoints, heatmaps, offset_masks, Ks, wrist_depths in tqdm(val_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)
            Ks = to_device(Ks)
            wrist_depths = to_device(wrist_depths)

            # Get predictions for regression and heatmap paths
            regression_outputs, heatmap_outputs = model(inputs)
            loss, regression_loss, heatmap_loss, offset_loss = loss_func(
                regression_outputs, keypoints, heatmap_outputs, heatmaps, offset_masks
            )
            total_combined_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_offset_loss += offset_loss.item()

            all_preds.extend(regression_outputs.cpu().numpy().squeeze())
            all_labels.extend(keypoints.cpu().numpy())

            # Calculate mpjpe on batch
            batch_mpjpe = mpjpe_3D(regression_outputs, keypoints, Ks, wrist_depths, image_size)
            # Multiply by batch size to get total pjpe for the batch
            mpjpe += batch_mpjpe.item() * keypoints.shape[0]

            # Calculate 3D pck on batch
            batch_pck20 = pck_3D(regression_outputs, keypoints, 20, Ks, wrist_depths, image_size)
            pck20 += batch_pck20.item() * keypoints.shape[0]
            batch_pck40 = pck_3D(regression_outputs, keypoints, 40, Ks, wrist_depths, image_size)
            pck40 += batch_pck40.item() * keypoints.shape[0]


    # Divide by # of images
    mpjpe /= len(all_preds)
    pck20 /= len(all_preds)
    pck40 /= len(all_preds)

    average_val_loss = total_combined_loss / len(val_loader)
    average_val_regression_loss = total_regression_loss / len(val_loader)
    average_val_heatmap_loss = total_heatmap_loss / len(val_loader)
    average_val_offset_loss = total_offset_loss / len(val_loader)
    
    # Flatten
    all_preds_flattened = np.concatenate(all_preds, axis=0)
    all_labels_flattened = np.concatenate(all_labels, axis=0)

    preds_concat = torch.cat([torch.tensor(pred) for pred in all_preds_flattened])
    labels_concat = torch.cat([torch.tensor(label) for label in all_labels_flattened])

    mae = torch.mean(torch.abs(preds_concat - labels_concat)).item()

    # Reshape to [batch, num_keypoints, 3]
    preds_kp = preds_concat.view(-1, 21, 3)
    labels_kp = labels_concat.view(-1, 21, 3)

    # Calculate pck metrics
    # For FreiHAND: p1 = 9 (middle finger bottom), p2 = 12 (middle finger top) (not conventional)
    correct005 = pck_2D(preds_kp[..., :2], labels_kp[..., :2], 0.05, 9, 12)
    correct02 = pck_2D(preds_kp[..., :2], labels_kp[..., :2], 0.2, 9, 12)

    pck005 = correct005.item()
    pck02 = correct02.item()

    metrics = {
        "mae": mae,
        "pck@0.05": pck005,
        "pck@0.2": pck02,
        "pck@20mm": pck20,
        "pck@40mm": pck40,
        "mpjpe": mpjpe,
        "average_val_loss": average_val_loss,
        "average_val_regression_loss": average_val_regression_loss,
        "average_val_heatmap_loss": average_val_heatmap_loss,
        "average_val_offset_loss": average_val_offset_loss
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
        unfreeze_epoch=40,
        image_size=224,
        runs_dir="models/BlazePoseFreiHAND/runs",
    ):
    # create log file
    time = str(datetime.now())
    os.mkdir(runs_dir + "/" + time)
    logfile_path = runs_dir + "/" + time + "/metrics.csv"
    best_pck005 = 0

    # training loop
    for i in range(start_epoch, num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        if i == unfreeze_epoch:
            print("Unfreezing initial layer(s)")
            for param in model.bb1.parameters():
                param.requires_grad = True

        total_combined_loss = 0.0
        total_regression_loss = 0.0
        total_heatmap_loss = 0.0
        total_offset_loss = 0.0

        model.train()
        for inputs, keypoints, heatmaps, offset_masks, Ks, wrist_depths in tqdm(train_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)

            optimizer.zero_grad()

            # Get predictions for regression and heatmap paths
            regression_outputs, heatmap_outputs = model(inputs)

            loss, regression_loss, heatmap_loss, offset_loss = loss_func(
                regression_outputs, keypoints, heatmap_outputs, heatmaps, offset_masks
            )
            loss.backward()
            optimizer.step()

            total_combined_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_heatmap_loss += heatmap_loss.item()
            total_offset_loss += offset_loss.item()

        # print and log metrics
        average_train_loss = total_combined_loss / len(train_loader)
        average_train_regression_loss = total_regression_loss / len(train_loader)
        average_train_heatmap_loss = total_heatmap_loss / len(train_loader)
        average_train_offset_loss = total_offset_loss / len(train_loader)

        metrics = validate(model, val_loader, loss_func, image_size)
        metrics["average_train_loss"] = average_train_loss
        metrics["average_train_regression_loss"] = average_train_regression_loss
        metrics["average_train_heatmap_loss"] = average_train_heatmap_loss
        metrics["average_train_offset_loss"] = average_train_offset_loss


        print(f'Epoch {i+1} Results:')
        print(f'Train Loss: {average_train_loss} | Regression: {average_train_regression_loss}'
             f' | Heatmap: {average_train_heatmap_loss} | Offset: {average_train_offset_loss}')
        print(f'Val Loss:   {metrics["average_val_loss"]} | Regression: {metrics["average_val_regression_loss"]}'
             f' | Heatmap: {metrics["average_val_heatmap_loss"]} | Offset: {metrics["average_val_offset_loss"]}')
        print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
        print(f'PCK@20mm: {metrics["pck@20mm"]}\tPCK@40mm: {metrics["pck@40mm"]}')
        print(f'MPJPE: {metrics["mpjpe"]}')

        log_results(logfile_path, metrics)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # save best model
        checkpoint = {
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        # Save best model best on pck@0.05
        pck005 = metrics['pck@0.05']
        if pck005 > best_pck005:
            torch.save(checkpoint, runs_dir + "/" + time + "/best.pt")
            best_pck005 = pck005

        torch.save(checkpoint, runs_dir + "/" + time + "/last.pt")

    # test model and print/log testing metrics
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func, image_size)
    print("Testing Results")
    print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
    print(f'PCK@20mm: {metrics["pck@20mm"]}\tPCK@40mm: {metrics["pck@40mm"]}')
    print(f'MPJPE: {metrics["mpjpe"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')

    test_logfile_path = runs_dir + "/" + time + "/test_metrics.csv"
    log_results(test_logfile_path, metrics)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch']

