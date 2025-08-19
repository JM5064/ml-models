import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from mpii_dataset import MPIIDataset
from PIL import Image


def visualize_dataset(dataset, num_samples=100):
    """
    Visualize samples from a PyTorch dataset that returns (image, keypoints),
    where keypoints are in format [[x, y, v], ...].
    """
    for i in range(min(num_samples, len(dataset))):
        image, keypoints, _ = dataset[i]

        # Convert PIL image to numpy
        image = np.array(image)

        # Draw keypoints
        for (x, y, v) in keypoints:
            if v == 1:
                # Occluded
                color = (0, 255, 0)
            else:
                # Visible
                color = (255, 0, 0)

            cv2.circle(image, (int(x), int(y)), 3, color, -1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Dataset', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def visualize_heatmaps(dataset, num_samples=100):
    """Visualize heatmaps for each image
    """
    for i in range(min(num_samples, len(dataset))):
        image, keypoints, heatmaps = dataset[i]
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Combine heatmaps
        combined_heatmap = np.zeros(heatmaps[0].shape)
        for heatmap in heatmaps:
            combined_heatmap += heatmap

        cv2.imshow('Image', image)
        cv2.imshow('Heatmap', combined_heatmap)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


mpii_dataset = MPIIDataset('datasets/MPII/mpii/images', 'datasets/MPII/mpii/annotations.json')
# visualize_dataset(mpii_dataset)
visualize_heatmaps(mpii_dataset)