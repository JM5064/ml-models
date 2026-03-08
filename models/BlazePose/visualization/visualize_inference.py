"""Usage:
python -m models.BlazePose.visualization.visualize_inference
"""

import numpy as np
import cv2
from PIL import Image
import time

import torch
from torchvision.transforms import v2

from models.BlazePose.blazepose import BlazePose
from datasets.MPII.mpii_dataset import MPIIDataset
from datasets.FreiHAND.visualize_dataloader import add_keypoints, add_heatmap_offsets


keypoint_map = {
    0: 'RF',    # right foot
    1: 'RK',    # right knee
    2: 'RH',    # right hip
    3: 'LH',    # left hip
    4: 'LK',    # left knee
    5: 'LF',    # left foot
    6: 'P',     # pelvis
    7: 'T',     # thorax
    8: 'N',     # upper neck
    9: 'H',     # head top
    10: 'RW',   # right wrist
    11: 'RE',   # right elbow
    12: 'RS',   # right shoulder
    13: 'LS',   # left shoulder
    14: 'LE',   # left elbow
    15: 'LW',   # left wrist
}


def load_model(model_path, device, num_keypoints=16):
    print("Using device", device)

    model = BlazePose(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def denormalize_color(tensor, mean, std):
    """Convert normalized color back into regular rgb"""
    return tensor * std + mean


def inference(model, dataset):
    """Run model inference and visualize keypoint, heatmap, and offset results"""
    for image, _, _, _ in dataset:
        # Evaluate model
        with torch.no_grad():
            input_tensor = torch.tensor(image).unsqueeze(0)

            keypoint_predictions, heatmap_offset_predictions = model(input_tensor)

            keypoint_predictions = np.array(keypoint_predictions.squeeze())
            heatmap_offset_predictions = np.array(heatmap_offset_predictions.squeeze())

        # Convert PIL image to numpy
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))

        # Recolor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = denormalize_color(image, mean=[0.472, 0.450, 0.413], std=[0.277, 0.272, 0.273])

        # Create keypoint image
        keypoints_image = add_keypoints(image, keypoint_predictions, keypoint_map)

        # Create heatmap and offset images
        heatmap_image, x_offset_image, y_offset_image = add_heatmap_offsets(heatmap_offset_predictions)

        cv2.imshow("Keypoints", keypoints_image)
        cv2.imshow("Heatmaps", heatmap_image)
        cv2.imshow("x offsets", x_offset_image)
        cv2.imshow("y offsets", y_offset_image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "models/BlazePose/runs/2026-03-08 17:39:18.053314/best.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(model_path, device)

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    test_dataset = mpii_dataset = MPIIDataset('datasets/MPII/mpii/images', 'datasets/MPII/mpii/test.json', transform=transform)
    
    inference(model, test_dataset)