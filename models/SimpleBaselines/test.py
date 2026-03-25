"""Usage:
python -m models.SimpleBaselines.test
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from tqdm import tqdm

from models.SimpleBaselines.model import SimpleBaselines
from models.utils import DEVICE
from datasets.MPII.mpii_dataset import MPIIDataset
from models.SimpleBaselines.losses.heatmap_loss import HeatmapLoss

from datasets.MPII.heatmap_inference import heatmap_inference, heatmap_inference_testing
from metrics.pck import pck_2D_visibile



def validate(model, val_loader, loss_func):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss= 0.0
    
    with torch.no_grad():
        for inputs, keypoints, heatmaps, offset_masks in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            offset_masks = offset_masks.to(DEVICE)

            # Get predictions for normal image, and flipped image
            heatmap_outputs = model(inputs)
            heatmap_outputs_flipped = model(torch.flip(inputs, dims=[-1]))
            loss = loss_func(heatmap_outputs, heatmaps, keypoints)
            total_loss += loss.item()

            keypoint_predictions = heatmap_inference_testing(heatmap_outputs, heatmap_outputs_flipped)

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


def test(model, test_loader, loss_func):
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'Test Loss:   {metrics["average_val_loss"]}')
    print(f'PCK@0.05: {metrics["pck@0.05"]}\tPCK@0.2: {metrics["pck@0.2"]}\tPCKh@0.5: {metrics["pckh@0.5"]}')


def load_model(model_path, num_keypoints=16):
    model = SimpleBaselines(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(DEVICE)

    model.eval()

    return model


if __name__ == "__main__":
    model_path = "models/SimpleBaselines/runs/mpii_simplebaselines/best.pt"
    model = load_model(model_path)

    images_dir = 'datasets/MPII/mpii/images'
    test_json = 'datasets/MPII/mpii/test.json'

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = MPIIDataset(images_dir, test_json, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    
    test(model, test_loader, HeatmapLoss())
