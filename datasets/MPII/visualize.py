import scipy.io
import cv2
import numpy as np
import os
import random
from PIL import Image


images_dir = 'datasets/MPII/mpii/images'
annotations_path = 'datasets/MPII/mpii/mpii_human_pose_v1_u12_1.mat'

mat = scipy.io.loadmat(annotations_path)

# dict_keys(['_fieldnames', 'annolist', 'img_train', 'version', 'single_person', 'act', 'video_list'])
records = mat['RELEASE'][0][0]
annolist = records['annolist'][0]
img_train = records['img_train'][0]
single_person = records['single_person']

keypoint_map = {
    0: 'RF',    # right foot
    1: 'RK',    # right knee
    2: 'RH',    # right hip
    3: 'LH',    # left hip
    4: 'LK',    # left knee
    5: 'LF',    # left foot
    6: 'P',     # pelvis
    7: 'T',     # thorax
    8: 'N',     # upper neck
    9: 'H',     # head top
    10: 'RW',   # right wrist
    11: 'RE',   # right elbow
    12: 'RS',   # right shoulder
    13: 'LS',   # left shoulder
    14: 'LE',   # left elbow
    15: 'LW',   # left wrist
}


def create_bounding_box(points, image_width, image_height):
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

    # Create initial padding bbox for the person based on their pose points
    padding = 0.35

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
    
    padded_width = max_x - min_x
    padded_height = max_y - min_y

    # For the current shorter side, extend them so that it's halfway between where it is now and the longer side
    max_length = max(padded_width, padded_height)

    x_extension = (max_length - padded_width) / 4
    y_extension = (max_length - padded_height) / 4

    min_x -= x_extension
    max_x += x_extension
    min_y -= y_extension
    min_y += y_extension

    return round(min_x), round(min_y), round(max_x), round(max_y)


def visualize_images():
    start_image = random.randint(0, 24884)
    start_image = 0
    end_image = min(start_image + 100, 24984)
    print(f'Viewing images {start_image} to {end_image}')
    for i in range(start_image, end_image):
        image_name = annolist[i]['image']['name'][0][0][0]
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print("Uh oh, missing image", image_path)
            continue

        image = cv2.imread(image_path)

        if image is None:
            print("Error reading image", image_path)
            continue

        if img_train[i] == 0:
            continue

        height, width, c = image.shape

        print("----")
        print('Single person:', single_person[i][0])
        print('Image train:', img_train[i])

        if annolist[i]['annorect'] is None or len(annolist[i]['annorect']) == 0:
            print("sus, no annorect", annolist[i]['annorect'], annolist[i])
            continue

        # Put all people ids that are sufficiently separated in a set
        individuals_ids = set()
        for sp in single_person[i][0]:
            individuals_ids.add(int(np.squeeze(sp[0])) - 1)

        annorects = annolist[i]['annorect'][0]
        
        for ridx in range(len(annorects)):
            if annorects[ridx] is None or 'annopoints' not in annorects[ridx].dtype.names or annorects[ridx]['annopoints'].size == 0:
                print("No annotations")
                continue

            annopoints = annorects[ridx]['annopoints'][0][0]
            
            if annopoints.size == 0 or 'point' not in annopoints.dtype.names:
                print('Annotations exist, but not really')
                continue

            if ridx in individuals_ids:
                color = (0, 0, 255)
            else:
                color = (100, 100, 100)

            points = annopoints['point'][0]

            x1, y1, x2, y2 = create_bounding_box(points, width, height)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            for point in points:
                x = int(np.squeeze(point['x']))
                y = int(np.squeeze(point['y']))
                visible = point['is_visible']
                keypoint_id = np.squeeze(point['id'])

                color = (0, 0 ,0)

                # Draw keypoint
                if len(visible) == 0 or str(np.squeeze(visible)) == '1':
                    # Visible
                    color = (0, 255, 0)
                elif str(np.squeeze(visible)) == '0':
                    # Occluded
                    color = (0, 0, 255)
                else:
                    print("new visible type:", visible)

                cv2.circle(image, (x, y), 3, color, -1)
                cv2.putText(image, keypoint_map[int(keypoint_id)], (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)

        cv2.imshow('Image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


visualize_images()

print("done")