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

    def __init__(self, images_dir, annotations_json, transform=None, output_size=256, heatmap_size=64):
        self.images_dir = images_dir

        self.dataset_json = json.load(open(annotations_json, "r"))
        self.data = []
        for image_name in self.dataset_json:
            for person in self.dataset_json[image_name]:
                self.data.append((image_name, person))

        self.transform = transform
        self.output_size = output_size
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
        np_image = np.array(letterbox_image)

        # Normalize keypoints
        normalized_keypoints = self.normalize_keypoints(letterbox_keypoints)
        
        # Convert keypoints to tensor
        tensor_keypoints = torch.tensor(normalized_keypoints, dtype=torch.float32)

        # Create heatmaps / offset maps
        heatmaps, offset_masks = self.create_heatmap_and_offset_maps(letterbox_keypoints)

        # Apply transformations
        if self.transform:
            np_image = self.transform(np_image)

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
        scale = self.output_size / longest_side

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

        x_pad_left = (self.output_size - new_width) // 2
        y_pad_top = (self.output_size - new_height) // 2

        # Add lettercrop padding to scaled image
        lettercrop_image = Image.new(scaled_image.mode, (self.output_size, self.output_size), (0, 0, 0))
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

            normalized_keypoints.append([keypoint[0] / self.output_size, keypoint[1] / self.output_size, keypoint[2]])

        return normalized_keypoints

    
    def create_heatmap_channel(self, x, y, sigma=3):
        rows = np.arange(self.output_size, dtype=np.float32)[:, None] # Column vector (height, 1)
        cols = np.arange(self.output_size, dtype=np.float32)[None, :] # Row vector (1, width)

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
        
        x_heatmap_true_center = (x / self.output_size) * self.heatmap_size
        y_heatmap_true_center = (y / self.output_size) * self.heatmap_size

        rows = np.arange(self.heatmap_size, dtype=np.float32)[:, None] # Column vector (height, 1)
        cols = np.arange(self.heatmap_size, dtype=np.float32)[None, :] # Row vector (1, width)

        x_offset_map = (x_heatmap_true_center - cols) * (heatmap_channel > threshold)
        y_offset_map = (y_heatmap_true_center - rows) * (heatmap_channel > threshold)

        mask = (heatmap_channel > threshold).astype(np.float32)

        return x_offset_map, y_offset_map, mask


    def create_heatmap_and_offset_maps(self, letterbox_keypoints):        
        heatmaps = []
        x_offset_maps = []
        y_offset_maps = []
        offset_masks = []

        for keypoint in letterbox_keypoints:
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


