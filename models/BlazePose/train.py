import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch


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
    file.flush() # Makes file update immediately


def calculate_blazepose_pck(preds_kp, labels_kp, percent):
    """Calculate pck for keypoints. A keypoint is considered correct if it's within percent% of the torso size
    args:
        preds_kp: [batch, num_keypoints, 2]
        labels_kp: [batch, num_keypoints, 2]
        percent: number, percentage for pck threshold

    returns:
        pck: % correct keypoints according to threshold    
    """

    # 2: right hip, 3: left hip, 12: right shoulder, 13: left shoulder
    left_hip_labels = labels_kp[:, 3, :]
    right_shoulder_labels = labels_kp[:, 12, :]

    torso_size_labels = torch.norm(left_hip_labels - right_shoulder_labels, dim=1)

    # Calculate distances between predicted keypoints and labels
    distances = torch.norm(preds_kp - labels_kp, dim=2)

    # Normalize distances wrt torso size instead of image size
    torso_distances = distances / torso_size_labels[:, None]

    # Count as correct if the distance is within pck% of the torso size
    correct = (torso_distances < percent).float()

    # Make sure that not labeled points (-1) are not included in the correctness calculation (use x for the visibility)
    visibilities = labels_kp[:, :, 0]
    mask = (visibilities != -1).float()

    # Mask out images where the torso doesn't exist / is too small
    valid_torsos_mask = (torso_distances > 0.01).float()
    mask = mask * valid_torsos_mask

    correct = correct * mask

    # Calculate pck as the number of correct keypoints over the total number of valid keypoints
    pck = correct.sum() / mask.sum()

    return pck


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
    correct005 = calculate_blazepose_pck(preds_kp, labels_kp, 0.05)
    correct02 = calculate_blazepose_pck(preds_kp, labels_kp, 0.2)

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
        runs_dir="models/BlazePose/runs",
    ):
    # create log file
    time = str(datetime.now())
    os.mkdir(runs_dir + "/" + time)
    logfile = open(runs_dir + "/" + time + "/metrics.txt", "a")
    best_pck005 = 0

    # training loop
    for i in range(start_epoch, num_epochs):
        print(f'Epoch {i+1}/{num_epochs}')

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

        log_results(logfile, metrics)

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

    test_logfile = open(runs_dir + "/" + time + "/test_metrics.txt", "a")
    log_results(test_logfile, metrics)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch']

