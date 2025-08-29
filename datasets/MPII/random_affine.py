import numpy as np
from PIL import Image
import cv2
import random
import time

import torch.nn as nn
from torchvision.transforms import v2


class RandomAffine(nn.Module):

    def __init__(self, 
        degrees: float | None, 
        translate: tuple[float, float] | None, 
        scale: tuple[float, float] | None,
        shear: float | None,
        image_size=256
    ):
        """Applies an affine transformation onto the image and keypoints
        args:
            degrees: maximum rotation in degrees
            translate: (max_x_fraction, max_y_fraction)
            scale: (min_scale, max_scale)
            shear: maximum sheer magnitude
        """

        super().__init__()

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        self.image_size = image_size


    def forward(self, image, keypoints):
        transformed_image, transformed_keypoints = self.random_affine(image, keypoints)

        return transformed_image, transformed_keypoints


    def random_affine(self, image, keypoints):
        """Applies an affine transformation onto the image and keypoints
        args:
            image: PIL image
            keypoints: [[x1, y1, v1], ...] keypoints
        
        returns:
            transformed_image: PIL image after random affine transformation
            keypoints: keypoints after random affine transformation
        """

        identity = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.copy(identity)
        translation_matrix = np.copy(identity)
        scale_matrix = np.copy(identity)
        shear_matrix = np.copy(identity)

        # For making sure the transformations are applied to the center of the image
        center_translation = np.array([
            [1, 0, -self.image_size // 2],
            [0, 1, -self.image_size // 2],
            [0, 0, 1]
        ])

        center_translation_back = np.array([
            [1, 0, self.image_size // 2],
            [0, 1, self.image_size // 2],
            [0, 0, 1]
        ])

        if self.degrees is not None:
            """Rotation affine matrix:
            [[cosθ -sinθ 0]
             [sinθ cosθ  0]
             [0     0    1]]
            """

            rand_degree = random.uniform(-self.degrees, self.degrees)

            # Convert to radians
            rand_degree = rand_degree * np.pi / 180

            rotation_matrix = np.array([
                [np.cos(rand_degree), -np.sin(rand_degree), 0],
                [np.sin(rand_degree), np.cos(rand_degree), 0],
                [0, 0, 1]
            ])

            rotation_matrix = center_translation_back @ rotation_matrix @ center_translation

        if self.translate is not None:
            """Translate affine matrix:
            [[1 0 tx]
             [0 1 ty]
             [0 0 1 ]]
            """

            max_x_fraction, max_y_fraction = self.translate

            x_fraction = random.uniform(-max_x_fraction, max_x_fraction)
            y_fraction = random.uniform(-max_y_fraction, max_y_fraction)

            x_translate = self.image_size * x_fraction
            y_translate = self.image_size * y_fraction

            translation_matrix = np.array([
                [1, 0, x_translate],
                [0, 1, y_translate],
                [0, 0, 1]
            ])

        
        if self.scale is not None:
            """Scale affine matrix:
            [[sx 0 0]
             [0 sy 0]
             [0 0  1]]
            """

            rand_scale = random.uniform(self.scale[0], self.scale[1])

            scale_matrix = np.array([
                [rand_scale, 0, 0],
                [0, rand_scale, 0],
                [0, 0, 1]
            ])

            scale_matrix = center_translation_back @ scale_matrix @ center_translation
        

        if self.shear is not None:
            """Shear affine matrix:
            [[1 s 0]
             [0 1 0]
             [0 0 1]]
            """

            rand_shear = random.uniform(-self.shear, self.shear)

            shear_matrix = np.array([
                [1, rand_shear, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

            shear_matrix = center_translation_back @ shear_matrix @ center_translation


        transformation_matrix = translation_matrix @ rotation_matrix @ scale_matrix @ shear_matrix

        # PIL's affine transform inverts the matrix for some reason
        inverted_transformation_matrix = np.linalg.inv(transformation_matrix)

        a = inverted_transformation_matrix[0][0]
        b = inverted_transformation_matrix[0][1]
        c = inverted_transformation_matrix[0][2]
        d = inverted_transformation_matrix[1][0]
        e = inverted_transformation_matrix[1][1]
        f = inverted_transformation_matrix[1][2]

        # Apply transformation to image
        transformed_image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)

        # Apply transformation to keypoints
        transformed_keypoints = self.transform_keypoints(keypoints, transformation_matrix)

        return transformed_image, transformed_keypoints


    def transform_keypoints(self, keypoints, transformation_matrix):
        """Transforms keypoints according to transformation matrix
        args:
            keypoints: [[x1, y1, v1], ...] keypoints
            transformation_matrix: 3x3 numpy affine transformation matrix

        returns:
            transformed_keypoints: keypoints after affine transformation
        """

        transformed_keypoints = []
        for keypoint in keypoints:
            if keypoint[0] == -1: 
                transformed_keypoints.append([-1, -1, -1])
                continue

            homogeneous_keypoint = np.array([
                [keypoint[0]],
                [keypoint[1]],
                [1]
            ])

            transformed_keypoint = transformation_matrix @ homogeneous_keypoint

            new_x = float(transformed_keypoint[0])
            new_y = float(transformed_keypoint[1])

            # Mark as not labeled if transformed keypoint lands outside the image
            if new_x < 0 or new_x > self.image_size or new_y < 0 or new_y > self.image_size:
                transformed_keypoints.append([-1, -1, -1])
                continue

            transformed_keypoints.append([new_x, new_y, keypoint[2]])

        return transformed_keypoints

