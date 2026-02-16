"""
FreiHAND dataset with heatmaps, x, y offset maps, and keypoints
"""

import os
import math
import numpy as np
import random
import cv2
import jsonyx as json
import time

import torch
from PIL import Image
from torch.utils.data import Dataset


class FreiHAND(Dataset):

    def __init__(
        self,
        images_dir, 
        keypoints_json, 
        intrinsics_json,
        scale_json,
        transform=None, 
        image_size=224, 
        heatmap_size=56
    ):
        self.images_dir = images_dir
        self.image_names = sorted(os.listdir(images_dir))

        self.keypoints_json = json.load(open(keypoints_json, "r"))
        self.intrinsics_json = json.load(open(intrinsics_json, "r"))
        self.scale_json = json.load(open(scale_json, "r"))

        # data contains (image name, xyz keypoints, intrinsics matrix K) for each image
        self.data = []
        for i in range(len(self.image_names)):
            self.data.append(
                (
                    self.image_names[i], 
                    # training images are duplicated 4 times with differing backgrounds
                    np.array(self.keypoints_json[i % len(self.keypoints_json)]),
                    np.array(self.intrinsics_json[i % len(self.intrinsics_json)])
                )
            )

        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = heatmap_size


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        """Get a sample from the dataset by index
        args:
            index: number

        returns:
            np_image: ndarray of the input image
            tensor_keypoints: [[x1, y1, z1], ...] keypoints
            heatmaps: ndarray of the heatmaps
            offset_masks: ndarray of offset masks
        """

        image_name, keypoints, K = self.data[index]
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path)

        # Apply transformations
        if self.transform:
            image, keypoints = self.transform(image, keypoints)
            keypoints = np.array(keypoints).squeeze()

        # Convert image to numpy
        np_image = np.array(image)

        # Project xyZ keypoints
        projected_keypoints = self.project_keypoints(keypoints, K)

        # Normalize keypoints
        normalized_keypoints = self.normalize_keypoints(projected_keypoints)
        
        # Convert keypoints to tensor
        tensor_keypoints = torch.tensor(normalized_keypoints, dtype=torch.float32)

        # Create heatmaps / offset maps
        heatmaps, offset_masks = self.create_heatmap_and_offset_maps(projected_keypoints)

        return np_image, tensor_keypoints, heatmaps, offset_masks


    def project_keypoints(self, keypoints, K):
        """Project 3D keypoints onto 2D image space, leaving Z coordinate the same
        args:
            keypoints: np.array([[X1, Y1, Z1], ...])
            K: np.array([3D intrinsics matrix])
        
        returns:
            keypoints_xyZ: np.array([[x1, y1, Z1], ...])
        """

        # Project keypoints
        keypoints_xyZ = (keypoints @ np.transpose(K)) / keypoints[:, 2:3]
        keypoints_xyZ[:, 2] = keypoints[:, 2]

        return keypoints_xyZ
    

    def normalize_depths(self, keypoints):
        """Normalize z coordinate with respect to the wrist
        args:
            keypoints: np.array([[X1, Y1, Z1], ...])

        returns:
            normalized_depths: np.array([z1, z2, ...])
        """

        depths = keypoints[:, 2]

        # Normalize wrt wrist (keypoint 0)
        wrist_depth = depths[0]

        normalized_depths = depths - wrist_depth

        return normalized_depths

        
    def normalize_keypoints(self, keypoints):
        """Normalizes keypoints between 0-1
        args:
            keypoints: np.array([[x1, y1, Z1], ...])

        returns:
            normalized_keypoints: [[x1', y1', z1'], ...] normalized keypoints
        """
        
        normalized_keypoints = keypoints.copy()

        # Normalize xy
        normalized_keypoints[:, :2] /= self.image_size

        # Normalize z
        normalized_keypoints[:, 2] = self.normalize_depths(normalized_keypoints)

        return normalized_keypoints
    

    def create_heatmap_and_offset_maps(self, keypoints, sigma=1.0, radius_threshold=5):
        """
        args:
            keypoints: np.array([[x1, y1, Z1], ...])
            sigma: standard deviation for Gaussian heatmap
            radius_threshold: Max distance for valid offsets

        returns:
            heatmap_and_offset_maps: np.array of concatenated heatmap and offset maps
            offset_masks: np.array offset mask for masking offset losses
        """
        num_keypoints = len(keypoints)

        # Create coordinate grid
        x_grid = np.arange(self.heatmap_size)
        y_grid = np.arange(self.heatmap_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Scale keypoints down to heatmap size
        scale = self.heatmap_size / self.image_size     
        scaled_keypoints = keypoints * scale

        # Reshape keypoints for broadcasting
        xs = scaled_keypoints[:, 0].reshape(num_keypoints, 1, 1)
        ys = scaled_keypoints[:, 1].reshape(num_keypoints, 1, 1)

        # Calculate Offsets
        x_offsets = xs - xx
        y_offsets = ys - yy

        # Calculate squared distances
        dist_sq = x_offsets ** 2 + y_offsets ** 2

        # Create heatmaps
        heatmaps = np.exp(-dist_sq / (2 * sigma ** 2))

        # Generate offset masks
        offset_masks = (dist_sq <= radius_threshold ** 2).astype(np.float32)

        # Apply offset masks
        x_offset_maps = x_offsets * offset_masks
        y_offset_maps = y_offsets * offset_masks

        # Combine heatmaps and offset maps
        heatmap_and_offset_maps = np.concatenate([heatmaps, x_offset_maps, y_offset_maps], axis=0).astype(np.float32)

        return heatmap_and_offset_maps, offset_masks


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'

    transform = v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    freihand = FreiHAND(images_dir, keypoints_json, intrinsics_json, scale_json, transform=transform)

    dl = DataLoader(freihand, batch_size=1, shuffle=False)

    for item in dl:
        for i in range(len(item)):
            print(item[i])
            print("--------------------------------")

        break