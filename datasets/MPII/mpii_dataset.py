"""
MPII Human Pose dataset with heatmaps, x, y offset maps, and keypoints
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


class MPIIDataset(Dataset):

    def __init__(self, images_dir, annotations_json, transform=None, image_size=256, heatmap_size=64):
        self.images_dir = images_dir

        self.dataset_json = json.load(open(annotations_json, "r"))
        self.data = []
        for image_name in self.dataset_json:
            for person in self.dataset_json[image_name]:
                self.data.append((image_name, person))

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

        image_name, person = self.data[index]
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path)

        # Create letterbox crop for image
        letterbox_image, letterbox_keypoints = self.create_letterbox_image(image, person)

        # Apply transformations
        if self.transform:
            transformed_image, transformed_keypoints = self.transform(letterbox_image, letterbox_keypoints)

        # Convert image to numpy
        np_image = np.array(transformed_image)

        # Normalize keypoints
        normalized_keypoints = self.normalize_keypoints(transformed_keypoints)
        
        # Convert keypoints to tensor
        tensor_keypoints = torch.tensor(normalized_keypoints, dtype=torch.float32)

        # Create heatmaps / offset maps
        heatmaps, offset_masks = self.create_heatmap_and_offset_maps(transformed_keypoints)

        return np_image, tensor_keypoints, heatmaps, offset_masks


    def crop_person(self, image, person):
        """Crops image give the person's bbox
        args:
            image: PIL image
            person: { bbox: [x1, y1, x2, y2], keypoints: [[x1, y1, v1], ...]}

        returns:
            cropped_image: PIL image
            cropped_keypoints: [[x1, y1, v1], ...] keypoints, but correct to the crop
        """

        x1, y1, x2, y2 = person["bbox"]

        # Subtract top left from all keypoints
        cropped_keypoints = []
        for keypoint in person["keypoints"]:
            if keypoint[0] == -1: 
                cropped_keypoints.append([-1, -1, -1])
                continue

            cropped_keypoints.append([keypoint[0] - x1, keypoint[1] - y1, keypoint[2]])

        cropped_image = image.crop((x1, y1, x2, y2))

        return cropped_image, cropped_keypoints


    def create_letterbox_image(self, image, person):
        """Creates letterbox cropped image
        args:
            image: PIL image
            person: { bbox: [x1, y1, x2, y2], keypoints: [[x1, y1, v1], ...]}

        returns:
            letterbox_image: PIL image
            letterbox_keypoints: [[x1, y1, v1], ...] keypoints, but correct to the crop
        """

        x1, y1, x2, y2 = person["bbox"]
        width = x2 - x1
        height = y2 - y1

        # Crop image to person
        cropped_image, cropped_keypoints = self.crop_person(image, person)

        # Calculate scale factor
        longest_side = max(width, height)
        scale = self.image_size / longest_side

        new_width = round(width * scale)
        new_height = round(height * scale)

        # Resize image
        scaled_image = cropped_image.resize((new_width, new_height))

        # Scale keypoints
        scaled_keypoints = []
        for keypoint in cropped_keypoints:
            if keypoint[0] == -1: 
                scaled_keypoints.append([-1, -1, -1])
                continue
            
            scaled_keypoints.append([keypoint[0] * scale, keypoint[1] * scale, keypoint[2]])

        x_pad_left = (self.image_size - new_width) // 2
        y_pad_top = (self.image_size - new_height) // 2

        # Add lettercrop padding to scaled image
        lettercrop_image = Image.new(scaled_image.mode, (self.image_size, self.image_size), (0, 0, 0))
        lettercrop_image.paste(scaled_image, (x_pad_left, y_pad_top))

        # Add lettercrop padding to keypoints
        lettercrop_keypoints = []
        for keypoint in scaled_keypoints:
            if keypoint[0] == -1: 
                lettercrop_keypoints.append([-1, -1, -1])
                continue

            lettercrop_keypoints.append([keypoint[0] + x_pad_left, keypoint[1] + y_pad_top, keypoint[2]])

        return lettercrop_image, lettercrop_keypoints


    def normalize_keypoints(self, keypoints):
        """Normalizes keypoints between 0-1
        args:
            keypoints: [[x1, y1, v1], ...] keypoints

        returns:
            normalized_keypoints: [[x1, y1, v1], ...] normalized keypoints
        """
        
        normalized_keypoints = []
        for keypoint in keypoints:
            if keypoint[0] == -1: 
                normalized_keypoints.append([-1, -1, -1])
                continue

            normalized_keypoints.append([keypoint[0] / self.image_size, keypoint[1] / self.image_size, keypoint[2]])

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
        keypoints = np.array(keypoints)
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

