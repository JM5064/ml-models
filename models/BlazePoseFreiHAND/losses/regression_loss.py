import numpy as np

import torch
import torch.nn as nn


class RegressionLoss(nn.Module):

    def __init__(self, alpha=0.5):
        super().__init__()

        self.alpha = alpha


    def forward(self, preds, labels):
        if preds.shape != labels.shape:
            print("Uh oh, predictions and labels have differing shapes", preds.shape, labels.shape)

        # preds and labels have shape [batch_size, num_keypoints, 3]
        # ... -> keep batch_size, num_keypoints the same, take first two points of the last dimension
        preds_xy = preds[..., :2]
        labels_xy = labels[..., :2]

        preds_z = preds[..., 2]
        labels_z = labels[..., 2]


        # Calculate squared errors between keypoints
        squared_errors = (preds_xy - labels_xy) ** 2
        xy_loss = squared_errors.sum()


        # Calculate squared errors between z values
        squared_errors = (preds_z - labels_z) ** 2
        
        z_loss = squared_errors.sum()

        # Calculate total loss
        total_loss = xy_loss + z_loss * self.alpha

        # print("Coord loss:", coord_loss)
        # print("Visibility loss:", visibility_loss * self.alpha)

        return total_loss


if __name__ == "__main__":
    # Batch of 2 images, 5 keypoints each
    outputs = torch.tensor([
        [[10, 20, 0], [30, 40, 1], [50, 60, 0], [70, 80, 0], [90, 100, 0]],
        [[11, 21, 0], [31, 41, 1], [51, 61, 0], [71, 81, 0.9], [91, 101, 0.5]]
    ])

    labels = torch.tensor([
        [[10, 20, 0], [29, 41, -1], [49, 59, 0], [70, 79, 0], [88, 102, 0]],
        [[12, 22, 1], [29, 39, 1], [52, 62, -1], [70, 82, 0], [89, 99, 0]]
    ])

    criterion = RegressionLoss()

    loss = criterion(outputs, labels)
    print("Total loss:", loss)



