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
        heatmap_size=64
    ):
        self.images_dir = images_dir
        self.image_names = sorted(os.listdir(images_dir))

        self.keypoints_json = json.load(open(keypoints_json, "r"))
        self.intrinsics_json = json.load(open(intrinsics_json, "r"))
        self.scale_json = json.load(open(scale_json, "r"))

        self.data = []
        for i in range(len(self.image_names)):
            self.data.append((self.image_names[i], self.keypoints_json[i]))

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
            letterbox_keypoints: [[x1, y1, v1], ...] keypoints
            heatmaps: ndarray of the heatmaps
        """

        image_name, keypoints = self.data[index]
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path)

        # Apply transformations
        if self.transform:
            transformed_image, transformed_keypoints = self.transform(image, keypoints)

        # Convert image to numpy
        np_image = np.array(transformed_image)
        
        # Convert keypoints to tensor
        tensor_keypoints = torch.tensor(transformed_keypoints, dtype=torch.float32)

        # Create heatmaps / offset maps
        heatmaps, offset_masks = self.create_heatmap_and_offset_maps(transformed_keypoints)

        return np_image, tensor_keypoints, heatmaps, offset_masks


    def project_keypoints(self, keypoints, K):
        # Project and normalize keypoints
        keypoints_xy = np.array([self.project_keypoint(keypoint, K) for keypoint in keypoints])
        keypoints_xy = self.normalize_keypoints(keypoints_xy)

        # Normalize depth
        keypoints_z = self.normalize_depths(keypoints)

        return np.concatenate([keypoints_xy, keypoints_z])


    def project_keypoint(self, keypoint, K):
        """Projects 3D keypoint onto 2D image space
        args:
            keypoint: np.array([[X1, Y1, Z1], [X2, Y2, Z2], ...])
            K: np.array(3x3 intrinsics matrix)
        
        returns:
            xy: np.array([x, y])
        """

        xy = (K @ keypoint) / keypoint[2]
        xy = xy[:2]

        return xy
    

    def normalize_depths(self, keypoints):
        """Normalize z coordinate with respect to the wrist
        args:
            keypoints: np.array([[X1, Y1, Z1], [X2, Y2, Z2], ...])

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
            keypoints: [[x1, y1], ...] keypoints

        returns:
            normalized_keypoints: [[x1', y1'], ...] normalized keypoints
        """
        
        normalized_keypoints = keypoints / self.image_size

        return normalized_keypoints

    
    def create_heatmap_channel(self, x, y, sigma=1):
        rows = np.arange(self.image_size, dtype=np.float32)[:, None] # Column vector (height, 1)
        cols = np.arange(self.image_size, dtype=np.float32)[None, :] # Row vector (1, width)

        # Get squared distances from each point i, j to x, y
        dist_sq = (cols - x) ** 2 + (rows - y) ** 2 

        # Compute Gaussians
        channel = np.exp(-dist_sq / (2 * sigma ** 2))

        channel = cv2.resize(channel, (self.heatmap_size, self.heatmap_size), interpolation=cv2.INTER_LINEAR)
    
        return channel
    

    def create_offset_channel(self, x, y, heatmap_channel, threshold=1e-3):
        """Creates x and y offset channels for a given radius around the keypoint
        args:
            x, y: the x, y positions of the keypoint in image space (not normalized)
            threshold: a heatmap value must pass this threshold for its offset to be calcualted

        returns:
            x_offset_map: offset map in the x direction
            y_offset_map: offset map in the y direction
            mask: a mask signifying where the relevant offsets are
        """

        # Return None if keypoint not present
        if x == -1:
            return None, None, None
        
        x_heatmap_true_center = (x / self.image_size) * self.heatmap_size
        y_heatmap_true_center = (y / self.image_size) * self.heatmap_size

        rows = np.arange(self.heatmap_size, dtype=np.float32)[:, None] # Column vector (height, 1)
        cols = np.arange(self.heatmap_size, dtype=np.float32)[None, :] # Row vector (1, width)

        x_offset_map = (x_heatmap_true_center - cols) * (heatmap_channel > threshold)
        y_offset_map = (y_heatmap_true_center - rows) * (heatmap_channel > threshold)

        # Normalize offsets
        x_offset_map /= self.heatmap_size
        y_offset_map /= self.heatmap_size

        mask = (heatmap_channel > threshold).astype(np.float32)

        return x_offset_map, y_offset_map, mask


    def create_heatmap_and_offset_maps(self, keypoints):        
        heatmaps = []
        x_offset_maps = []
        y_offset_maps = []
        offset_masks = []

        for keypoint in keypoints:
            if keypoint[0] == -1:
                # Append blank heatmaps and offset maps if point not visible
                heatmaps.append(np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32))
                x_offset_maps.append(np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32))
                y_offset_maps.append(np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32))
                offset_masks.append(np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32))

                continue

            # Calculate heatmap and offset map for each keypoint
            heatmap_channel = self.create_heatmap_channel(keypoint[0], keypoint[1])
            heatmaps.append(heatmap_channel)

            x_offset_channel, y_offset_channel, offset_mask = self.create_offset_channel(keypoint[0], keypoint[1], heatmap_channel)

            x_offset_maps.append(x_offset_channel)
            y_offset_maps.append(y_offset_channel)
            offset_masks.append(offset_mask)

        # Combine heatmaps and offset maps
        heatmaps.extend(x_offset_maps)
        heatmaps.extend(y_offset_maps)

        return np.array(heatmaps), np.array(offset_masks)


