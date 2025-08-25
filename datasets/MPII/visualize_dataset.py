import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from mpii_dataset import MPIIDataset
from torchvision.transforms import v2
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
        image = np.transpose(image, (1, 2, 0))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw keypoints
        for (x, y, v) in keypoints:
            # Unnormalize coords
            x *= 256
            y *= 256

            if v == 1:
                # Occluded
                color = (0, 255, 0)
            else:
                # Visible
                color = (255, 0, 0)

            cv2.circle(image, (int(x), int(y)), 3, color, -1)

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
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        num_keypoints = heatmaps.shape[0] // 3

        # Combine heatmaps
        combined_heatmap = np.zeros(heatmaps[0].shape)
        for i in range(num_keypoints):
            combined_heatmap += heatmaps[i]

        # Combine x offset maps
        combined_x_offsets = np.zeros(heatmaps[0].shape)
        for i in range(num_keypoints, 2 * num_keypoints):
            combined_x_offsets += heatmaps[i]

        # Combine y offset maps
        combined_y_offsets = np.zeros(heatmaps[0].shape)
        for i in range(num_keypoints * 2, 3 * num_keypoints):
            combined_y_offsets += heatmaps[i]

        # Average them so values don't explode
        combined_x_offsets /= num_keypoints
        combined_y_offsets /= num_keypoints

        # Normalize for visualization
        norm_x = cv2.normalize(combined_x_offsets, None, 0, 255, cv2.NORM_MINMAX)
        norm_x = norm_x.astype(np.uint8)

        norm_y = cv2.normalize(combined_y_offsets, None, 0, 255, cv2.NORM_MINMAX)
        norm_y = norm_y.astype(np.uint8)

        # Apply color map for gradient visualization
        color_x = cv2.applyColorMap(norm_x, cv2.COLORMAP_JET)
        color_y = cv2.applyColorMap(norm_y, cv2.COLORMAP_JET)

        cv2.imshow('Image', image)
        cv2.imshow('Heatmap', combined_heatmap)
        cv2.imshow('x offset map', color_x)
        cv2.imshow('y offset map', color_y)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


train_transform = v2.Compose([
        v2.ToTensor(),
        # v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1)),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # v2.Normalize(mean=[0.472, 0.450, 0.413],
        #                     std=[0.277, 0.272, 0.273]),
    ])

mpii_dataset = MPIIDataset('datasets/MPII/mpii/images', 'datasets/MPII/mpii/annotations.json', transform=train_transform)
# visualize_dataset(mpii_dataset)
visualize_heatmaps(mpii_dataset)