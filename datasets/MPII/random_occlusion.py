import torch.nn as nn
from PIL import Image
import numpy as np
import random


class RandomOcclusion(nn.Module):

    def __init__(self, min_fraction, max_fraction, p, image_size=256, seed=None):
        super().__init__()

        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

        self.p = p
        self.image_size = image_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


    def forward(self, image, keypoints):
        rand = random.random()
        if rand > self.p:
            return image, keypoints
        
        # Generate random rectangle
        rect_width = int(random.uniform(self.min_fraction, self.max_fraction) * self.image_size)
        rect_height = int(random.uniform(self.min_fraction, self.max_fraction) * self.image_size)

        random_rectangle_image = self.generate_rectangle(rect_width, rect_height)

        # Place rectangle somewhere nearish the middle
        padding = int(self.image_size * 0.2)
        rect_x = random.randint(padding, max(self.image_size - rect_width - padding, padding))
        rect_y = random.randint(padding, max(self.image_size - rect_height - padding, padding))

        image.paste(random_rectangle_image, (rect_x, rect_y))

        # Update keypoint visibilities
        updated_keypoints = self.update_keypoints(keypoints, rect_x, rect_y, rect_width, rect_height)

        return image, updated_keypoints
        

    def generate_rectangle(self, rect_width, rect_height):
        # Generate rectangle with random colors
        random_rectangle = np.random.randint(0, 256, (rect_height, rect_width, 3), dtype=np.uint8)

        random_rectangle_image = Image.fromarray(random_rectangle, mode='RGB')

        return random_rectangle_image
    

    def update_keypoints(self, keypoints, rect_x, rect_y, rect_width, rect_height):
        """Update keypoint visibility to "occluded" if it lies within the rectangle
        args:
            keypoints: [[x1, y1, v1], ...] keypoints
            rect_x, rect_y: ints, top left corner of the rectangle
            rect_width, rect_height: ints, widths and heights of the rectangle

        returns:
            updated_keypoints: [[x1, y1, v1], ...] keypoints with visibilities updated
        """

        updated_keypoints = []
        for keypoint in keypoints:
            x, y, v = keypoint

            if x == -1: 
                updated_keypoints.append([-1, -1, -1])
                continue

            if rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height:
                # Change keypoint to occluded if it's inside the rectangle
                updated_keypoints.append([x, y, 0])
            else:
                updated_keypoints.append([x, y, v])

        return updated_keypoints






    


    