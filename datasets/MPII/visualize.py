import scipy.io
import cv2
import numpy as np
import os
import random


images_dir = 'datasets/MPII/mpii/images'
annotations_path = 'datasets/MPII/mpii/mpii_human_pose_v1_u12_1.mat'

mat = scipy.io.loadmat('datasets/MPII/mpii/mpii_human_pose_v1_u12_1.mat')

# dict_keys(['_fieldnames', 'annolist', 'img_train', 'version', 'single_person', 'act', 'video_list'])
records = mat['RELEASE'][0][0]
annolist = records['annolist'][0]
img_train = records['img_train'][0]

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


def visualize_images():
    start_image = random.randint(0, 24884)
    end_image = min(start_image + 100, 24984)
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

        if annolist[i]['annorect'] is None or len(annolist[i]['annorect']) == 0:
            print("sus, no annorect", annolist[i]['annorect'], annolist[i])
            continue

        annorects = annolist[i]['annorect'][0]
        
        for person_bbox in annorects:
            if person_bbox is None or 'annopoints' not in person_bbox.dtype.names or person_bbox['annopoints'].size == 0:
                print("No annotations")
                continue

            annopoints = person_bbox['annopoints'][0][0]
            
            if annopoints.size == 0 or 'point' not in annopoints.dtype.names:
                print('Annotations exist, but not really')
                continue

            points = annopoints['point'][0]
            print("Num keypoints:", len(points))
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

            print("----")

        cv2.imshow('Image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


visualize_images()

print("done")