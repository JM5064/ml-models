import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
from metrics.pck import pck_2D_visibile
from models.utils import to_device, log_results
from datasets.MPII.heatmap_inference import heatmap_inference


def validate(model, val_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss= 0.0
    
    with torch.no_grad():
        for inputs, keypoints, heatmaps, offset_masks in tqdm(val_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)

            # Get predictions for regression and heatmap paths
            heatmap_outputs = model(inputs)
            loss = loss_func(heatmap_outputs, heatmaps, keypoints)
            total_loss += loss.item()

            keypoint_predictions = heatmap_inference(heatmap_outputs)

            all_preds.extend(keypoint_predictions.cpu().numpy().squeeze())
            all_labels.extend(keypoints.cpu().numpy())

    average_val_loss = total_loss / len(val_loader)
    
    # Flatten
    all_preds_flattened = np.concatenate(all_preds, axis=0)
    all_labels_flattened = np.concatenate(all_labels, axis=0)

    preds_concat = torch.cat([torch.tensor(pred) for pred in all_preds_flattened])
    labels_concat = torch.cat([torch.tensor(label) for label in all_labels_flattened])

    # mae = torch.mean(torch.abs(preds_concat - labels_concat)).item()

    # Reshape to [batch, num_keypoints, 2]
    preds_kp = preds_concat.view(-1, 16, 2)
    labels_kp = labels_concat.view(-1, 16, 3)[:, :, :2]

    # Calculate pck metrics
    # For MPII: p1 = 3 (left hip), p2 = 12 (right shoulder) -> torso size
    pck005 = pck_2D_visibile(preds_kp, labels_kp, 0.05, 3, 12).item()
    pck02 = pck_2D_visibile(preds_kp, labels_kp, 0.2, 3, 12).item()

    # Normalizing wrt head size
    pckh05 = pck_2D_visibile(preds_kp, labels_kp, 0.5, 8, 9).item()

    metrics = {
        # "mae": mae,
        "pck@0.05": pck005,
        "pck@0.2": pck02,
        "pckh@0.5": pckh05,
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
        runs_dir="models/BlazePose/runs",
    ):
    log_directory = runs_dir
    # create log file for a new training session
    if start_epoch == 0:
        time = str(datetime.now())
        os.mkdir(runs_dir + "/" + time)
        log_directory = runs_dir + "/" + time
    best_pck005 = 0

    # training loop
    for i in range(start_epoch, num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

        total_loss = 0.0

        model.train()
        for inputs, keypoints, heatmaps, offset_masks in tqdm(train_loader):
            inputs = to_device(inputs)
            keypoints = to_device(keypoints)
            heatmaps = to_device(heatmaps)
            offset_masks = to_device(offset_masks)

            optimizer.zero_grad()

            # Get predictions for regression and heatmap paths
            heatmap_outputs = model(inputs)

            loss = loss_func(heatmap_outputs, heatmaps, keypoints)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print and log metrics
        average_train_loss = total_loss / len(train_loader)

        metrics = validate(model, val_loader, loss_func)
        metrics["average_train_loss"] = average_train_loss

        print(f'Epoch {i+1} Results:')

        print(f'Train Loss: {average_train_loss}')
        print(f'Val Loss:   {metrics["average_val_loss"]}')
        print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}\tPCKh@0.5: {metrics["pckh@0.5"]}')
        # print(f'MAE: {metrics["mae"]}')

        log_results(log_directory + "/metrics.csv", metrics)

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
            torch.save(checkpoint, log_directory + "/best.pt")
            best_pck005 = pck005

        # Save last model
        torch.save(checkpoint, log_directory + "/last.pt")

        # Save a model every 20 epochs
        if (i+1) % 20 == 0:
            torch.save(checkpoint, f'{log_directory}/epoch{i+1}.pt')


    # test model and print/log testing metrics
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'Test Loss:   {metrics["average_val_loss"]} | Regression: {metrics["average_val_regression_loss"]}'
            f' | Heatmap: {metrics["average_val_heatmap_loss"]} | Offset: {metrics["average_val_offset_loss"]}')
    print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}\tPCKh@0.5: {metrics["pckh@0.5"]}')
    # print(f'MAE: {metrics["mae"]}')

    test_logfile_path = log_directory + "/test_metrics.csv"
    log_results(test_logfile_path, metrics)
