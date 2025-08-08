"""
Convert mpii dataset mat file into custom json format:
[
  "00001.jpg": [
    {
      "bbox": [x, y, w, h],
      "keypoints": [[x1, y1, v1], ..., [x16, y16, v16]]
    },
    {
      "bbox": [x, y, w, h],
      "keypoints": [[x1, y1, v1], ..., [x16, y16, v16]]
    }
  ],
...
]
"""

import os
import jsonyx as json
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_annotations(annotations_path):
    mat = scipy.io.loadmat(annotations_path)

    records = mat['RELEASE'][0][0]

    annolist = records['annolist'][0]
    img_train = records['img_train'][0]
    single_person = records['single_person']

    return annolist, img_train, single_person


def create_annotations(annolist, img_train, single_person, images_dir, annotations_file_path):
    annotations = {}
    
    # Loop through entire dataset, creating annotations for each image
    for i in tqdm(range(24986)):
        # Skip testing images
        if img_train[i] == 0:
            continue

        # Skip images with no single people
        if len(single_person[i][0]) == 0:
            continue

        if annolist[i]['annorect'] is None or len(annolist[i]['annorect']) == 0:
            # print("sus, no annorect", annolist[i]['annorect'], annolist[i], i)
            continue

        image_name = annolist[i]['image']['name'][0][0][0]
        image_path = os.path.join(images_dir, image_name)

        with Image.open(image_path) as img:
            width, height = img.size

        # Put all people ids that are sufficiently separated in a set
        individuals_ids = set()
        for sp in single_person[i][0]:
            if len(sp) == 0: continue
            individuals_ids.add(int(np.squeeze(sp[0])) - 1)

        # Get the annorects for the current image
        annorects = annolist[i]['annorect'][0]

        # Store the annotations for every sufficiently separated person in the image
        person_annotations = []

        # Loop through each person (annorect) in the image
        for ridx in range(len(annorects)):
            if annorects[ridx] is None or 'annopoints' not in annorects[ridx].dtype.names or annorects[ridx]['annopoints'].size == 0:
                print("No annotations", i)
                continue

            # Skip if the current annorect person id isn't a sufficiently separated individual
            if ridx not in individuals_ids:
                continue

            # Create person annotation
            person_annotation = create_person_annotation(annorects[ridx], width, height)
            if person_annotation is None:
                continue

            person_annotations.append(person_annotation)

        if len(person_annotations) > 0:
            annotations[image_name] = person_annotations

    with open(annotations_file_path, 'w') as file:
            json.dump(annotations, file, indent=2, indent_leaves=False)

    print("Done")


def create_bounding_box(points, image_width, image_height):
    """Creates a bounding box over the person using their pose points
    
    return:
        tlx, tly, brx, bry
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for point in points:
        x = int(np.squeeze(point['x']))
        y = int(np.squeeze(point['y']))

        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    padding = 0.4

    width = max_x - min_x
    height = max_y - min_y

    min_x -= width * padding
    max_x += width * padding
    min_y -= height * padding
    max_y += height * padding

    min_x = max(min_x, 0)
    max_x = min(max_x, image_width)
    min_y = max(min_y, 0)
    max_y = min(max_y, image_height)

    return round(min_x), round(min_y), round(max_x), round(max_y)


def create_person_annotation(annorect, image_width, image_height):
    """
    Returns an object {
        "bbox": [x1, y1, x2, y2],
        "keypoints": [[x1, y1, v1], ..., [x16, y16, v16]]
    }
    """

    keypoints = [[-1, -1, -1] for _ in range(16)]

    annopoints = annorect['annopoints'][0][0]
    if annopoints.size == 0 or 'point' not in annopoints.dtype.names:
        print('Annotations exist, but not really')
        return None
    
    points = annopoints['point'][0]

    if len(points) < 5:
        print("Not enough points", points, len(points))
        return None

    for point in points:
        x = int(np.squeeze(point['x']))
        y = int(np.squeeze(point['y']))
        visible = point['is_visible']

        keypoint_id = int(np.squeeze(point['id']))

        # Format visible flag to 0 or 1
        visible_formatted = 0
        if len(visible) == 0 or str(np.squeeze(visible)) == '1':
            # Visible
            visible_formatted = 1
        elif str(np.squeeze(visible)) == '0':
            # Occluded
            visible_formatted = 0
        else:
            raise ValueError("new visible type:", visible, type(visible))
        
        if keypoints[keypoint_id][0] != -1:
            print("Keypoint id:", keypoint_id)
            print("Old Keypoint:", keypoints[keypoint_id])
            print("New keypoint:", [x, y, visible_formatted])
            print("Keypoints:", keypoints)

            print("Something went wrong, duplicate keypoint id?")
            return None

        # Set keypoint data
        keypoints[keypoint_id][0] = x
        keypoints[keypoint_id][1] = y
        keypoints[keypoint_id][2] = visible_formatted

        x1, y1, x2, y2 = create_bounding_box(points, image_width, image_height)

    
    return {
        "bbox": [x1, y1, x2, y2],
        "keypoints": keypoints
    }
        

def main():
    images_dir = 'datasets/MPII/mpii/images'
    annotations_path = 'datasets/MPII/mpii/mpii_human_pose_v1_u12_1.mat'
    annotations_json = 'datasets/MPII/mpii/annotations.json'

    annolist, img_train, single_person = load_annotations(annotations_path)
    create_annotations(annolist, img_train, single_person, images_dir, annotations_json)


if __name__ == "__main__":
    main()