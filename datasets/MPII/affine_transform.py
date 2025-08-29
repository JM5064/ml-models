import numpy as np
from PIL import Image
import cv2
import random
import time

import torch.nn as nn
from torchvision.transforms import v2


class AffineTransform:

    def __init__(self, seed=5064):
        # Run transformations over image and keypoints, then generate heatmap/offset maps
        random.seed(seed)


    def random_affine(self, 
        image,
        keypoints,
        degrees: float | None, 
        translate: tuple[float, float] | None, 
        scale: tuple[float, float] | None,
        shear: float | None
    ):
        """Applies an affine transformation onto the image and keypoints
        args:
            image: PIL image
            keypoints: [[x1, y1, v1], ...] keypoints
            degrees: maximum rotation in degrees
            translate: (max_x_fraction, max_y_fraction)
            scale: (min_scale, max_scale)
            shear: maximum sheer magnitude
        
        returns:
            transformed_image: PIL image after random affine transformation
            keypoints: keypoints after random affine transformation
        """

        image_size = image.size[0]

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
            [1, 0, -image_size // 2],
            [0, 1, -image_size // 2],
            [0, 0, 1]
        ])

        center_translation_back = np.array([
            [1, 0, image_size // 2],
            [0, 1, image_size // 2],
            [0, 0, 1]
        ])

        if degrees is not None:
            """Rotation affine matrix:
            [[cosθ -sinθ 0]
             [sinθ cosθ  0]
             [0     0    1]]
            """

            rand_degree = random.uniform(-degrees, degrees)

            # Convert to radians
            rand_degree = rand_degree * np.pi / 180

            rotation_matrix = np.array([
                [np.cos(rand_degree), -np.sin(rand_degree), 0],
                [np.sin(rand_degree), np.cos(rand_degree), 0],
                [0, 0, 1]
            ])

            rotation_matrix = center_translation_back @ rotation_matrix @ center_translation

        if translate is not None:
            """Translate affine matrix:
            [[1 0 tx]
             [0 1 ty]
             [0 0 1 ]]
            """

            max_x_fraction, max_y_fraction = translate

            x_fraction = random.uniform(-max_x_fraction, max_x_fraction)
            y_fraction = random.uniform(-max_y_fraction, max_y_fraction)

            x_translate = image_size * x_fraction
            y_translate = image_size * y_fraction

            translation_matrix = np.array([
                [1, 0, x_translate],
                [0, 1, y_translate],
                [0, 0, 1]
            ])

        
        if scale is not None:
            """Scale affine matrix:
            [[sx 0 0]
             [0 sy 0]
             [0 0  1]]
            """

            rand_scale = random.uniform(scale[0], scale[1])

            scale_matrix = np.array([
                [rand_scale, 0, 0],
                [0, rand_scale, 0],
                [0, 0, 1]
            ])

            scale_matrix = center_translation_back @ scale_matrix @ center_translation
        

        if shear is not None:
            """Shear affine matrix:
            [[1 s 0]
             [0 1 0]
             [0 0 1]]
            """

            rand_shear = random.uniform(-shear, shear)

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
        # TODO: mark keypoints which land outside the image after the transformation as not labeled?
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

            transformed_keypoints.append([float(transformed_keypoint[0]), float(transformed_keypoint[1]), keypoint[2]])


        return transformed_keypoints



if __name__ == "__main__":
    image = Image.open("datasets/MPII/ak.jpg")

    keypoints = [
          [493, 640, 1],
          [507, 507, 1],
          [503, 370, 1],
          [522, 370, 0],
          [540, 494, 0],
          [556, 617, 1],
          [513, 370, 0],
          [494, 191, 0],
          [508, 158, 1],
          [548, 70, 1],
          [528, 367, 1],
          [477, 311, 1],
          [472, 195, 1],
          [515, 186, 0],
          [526, 272, 0],
          [552, 325, 0]
        ]

    at = AffineTransform(seed=110)

    start = time.time()
    transformed_image, transformed_keypoints = at.random_affine(image, keypoints, 20, (0.1, 0.1), (0.8, 1.2), None)
    end = time.time()
    print(end - start)
    
    transformed_image = np.array(transformed_image)

    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", transformed_image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()