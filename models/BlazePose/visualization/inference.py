"""Usage:
python -m models.BlazePose.visualization.inference
"""

import numpy as np
import cv2
from PIL import Image
import time

import torch
from torchvision.transforms import v2

from ..blazepose import BlazePose
from datasets.MPII.mpii_dataset import MPIIDataset


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


def load_model(model_path, device, num_keypoints=16):
    print("Using device", device)

    model = BlazePose(num_keypoints)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def denormalize(tensor, mean, std):
    return tensor * std + mean


def inference(model, dataset):
    num_images = 100
    for i in range(min(num_images, len(dataset))):
        image, _, _, _ = dataset[i]

        # Evaluate model
        with torch.no_grad():
            input_tensor = torch.tensor(image).unsqueeze(0)

            keypoint_predictions, heatmap_offset_predictions = model(input_tensor)


        # Convert PIL image to numpy
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = denormalize(image, mean=[0.472, 0.450, 0.413], std=[0.277, 0.272, 0.273])

        for j in range(len(keypoint_predictions[0])):
                kp = keypoint_predictions[0][j]
                x, y, vis = kp

                center = (int(x * 256), int(y * 256))
                cv2.circle(image, center,  3, (0, 0, 255), -1)
                cv2.putText(image, keypoint_map[int(j)], (int(x * 256), int(y * 256)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                print(x.item(), y.item(), vis.item())
        print()

        cv2.imshow('Dataset', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    model_path = "models/BlazePose/runs/100_epochs/best.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(model_path, device)

    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    test_dataset = mpii_dataset = MPIIDataset('datasets/MPII/mpii/images', 'datasets/MPII/mpii/test.json', transform=transform)

    inference(model, test_dataset)