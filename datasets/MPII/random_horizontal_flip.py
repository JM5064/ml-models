import torch
import torch.nn as nn
from PIL import Image
import random


class RandomHorizontalFlip(nn.Module):

    def __init__(self, p, image_size=256, seed=None):
        super().__init__()

        self.p = p
        self.image_size = image_size

        if seed is not None:
            random.seed(seed)


    def forward(self, image, keypoints):
        rand = random.random()

        if rand > self.p:
            return image, keypoints
        
        flipped_image = self.flip_image(image)
        flipped_keypoints = self.flip_keypoints(keypoints)

        return flipped_image, flipped_keypoints


    def flip_image(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)


    def flip_keypoints(self, keypoints):
        midpoint = self.image_size // 2

        flipped_keypoints = []
        for keypoint in keypoints:
            if keypoint[0] == -1: 
                flipped_keypoints.append([-1, -1, -1])
                continue

            distance_to_midpoint = keypoint[0] - midpoint
            new_x = midpoint - distance_to_midpoint

            flipped_keypoints.append([new_x, keypoint[1], keypoint[2]])

        # Swap left <-> right keypoints
        flipped_keypoints[0], flipped_keypoints[5] = flipped_keypoints[5], flipped_keypoints[0] # Feet
        flipped_keypoints[1], flipped_keypoints[4] = flipped_keypoints[4], flipped_keypoints[1] # Knees
        flipped_keypoints[2], flipped_keypoints[3] = flipped_keypoints[3], flipped_keypoints[2] # Hips
        flipped_keypoints[10], flipped_keypoints[15] = flipped_keypoints[15], flipped_keypoints[10] # Wrists
        flipped_keypoints[11], flipped_keypoints[14] = flipped_keypoints[14], flipped_keypoints[11] # Elbows
        flipped_keypoints[12], flipped_keypoints[13] = flipped_keypoints[13], flipped_keypoints[12] # Shoulders

        return flipped_keypoints

