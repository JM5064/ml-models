import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
from metrics.pck import pck_2D_visibile
from models.utils import to_device, log_results


def validate(model, val_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, keypoints, heatmaps, offset_masks in tqdm(val_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)

            # Get predictions for regression and heatmap paths
            regression_outputs, heatmap_outputs = model(inputs)
            loss = loss_func(regression_outputs, keypoints, heatmap_outputs, heatmaps, offset_masks)
            running_loss += loss.item()

            all_preds.extend(regression_outputs.cpu().numpy().squeeze())
            all_labels.extend(keypoints.cpu().numpy())

    average_val_loss = running_loss / len(val_loader)
    
    
    # Flatten
    all_preds_flattened = np.concatenate(all_preds, axis=0)
    all_labels_flattened = np.concatenate(all_labels, axis=0)

    preds_concat = torch.cat([torch.tensor(pred) for pred in all_preds_flattened])
    labels_concat = torch.cat([torch.tensor(label) for label in all_labels_flattened])

    mae = torch.mean(torch.abs(preds_concat - labels_concat)).item()

    # Reshape to [batch, num_keypoints, 3], and grab just the x, y
    preds_kp = preds_concat.view(-1, 16, 3)[:, :, :2]
    labels_kp = labels_concat.view(-1, 16, 3)[:, :, :2]

    # Calculate pck metrics
    # For MPII: p1 = 3 (left hip), p2 = 12 (right shoulder) -> torso size
    correct005 = pck_2D_visibile(preds_kp, labels_kp, 0.05, 3, 12)
    correct02 = pck_2D_visibile(preds_kp, labels_kp, 0.2, 3, 12)

    pck005 = correct005.mean().item()
    pck02 = correct02.mean().item()

    metrics = {
        "mae": mae,
        "pck@0.05": pck005,
        "pck@0.2": pck02,
        "average_val_loss": average_val_loss,
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
        runs_dir="models/BlazePose/runs",
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

        model.train()
        running_loss = 0.0
        for inputs, keypoints, heatmaps, offset_masks in tqdm(train_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)

            optimizer.zero_grad()

            # Get predictions for regression and heatmap paths
            regression_outputs, heatmap_outputs = model(inputs)

            loss = loss_func(regression_outputs, keypoints, heatmap_outputs, heatmaps, offset_masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print and log metrics
        average_train_loss = running_loss / len(train_loader)
        metrics = validate(model, val_loader, loss_func)
        metrics["average_train_loss"] = average_train_loss


        print(f'Epoch {i+1} Results:')
        print(f'Train Loss: {average_train_loss}\tValidation Loss: {metrics["average_val_loss"]}')
        print(f'MAE: {metrics["mae"]}\tPCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')

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
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'MAE: {metrics["mae"]}\tPCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}')
    print(f'Test Loss: {metrics["average_val_loss"]}')

    test_logfile_path = runs_dir + "/" + time + "/test_metrics.csv"
    log_results(test_logfile_path, metrics)

