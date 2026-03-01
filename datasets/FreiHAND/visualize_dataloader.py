"""Visualizes keypoints and heatmaps from dataloader"""

import cv2
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from freihand_dataset import FreiHAND


def visualize_keypoints(dataset):
    for image, keypoints, _, _ in dataset:
        image = image.transpose(1, 2, 0) # transpose from 3 x 224 x 224 -> 224 x 224 x 3
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Unnormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        image = image * std + mean
        
        h, w, _ = image.shape
        
        # Unnormalize keypoints
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h

        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 1, (0, 0, 255), -1)

        cv2.imshow("Image", image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def visualize_heatmaps(dataset):
    """Visualize heatmaps for each image
    """
    for image, keypoints, heatmaps, offset_masks in dataset:
        image = image.transpose(1, 2, 0) # transpose from 3 x 224 x 224 -> 224 x 224 x 3
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Unnormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        image = image * std + mean

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


def main():
    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    vertices_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_verts.json'

    transform = v2.Compose([
        # RandomHorizontalFlip(0.5, seed=5064),
        # RandomAffine(degrees=25, translate=None, scale=(0.75, 1.25), shear=0.1, seed=5064),
        # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # RandomOcclusion(0.1, 0.3, 0.5, seed=5064),
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    dataset = FreiHAND(
        images_dir=images_dir, 
        keypoints_json=keypoints_path, 
        intrinsics_json=intrinsics_path,
        scale_json=scale_path,
        transform=transform)

    visualize_keypoints(dataset)
    visualize_heatmaps(dataset)

if __name__ == "__main__":
    main()