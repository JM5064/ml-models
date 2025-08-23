import numpy as np

import torch
import torch.nn as nn


class HeatmapLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, preds, heatmap_labels, regression_labels):
        if preds.shape != heatmap_labels.shape:
            print("Uh oh, heatmaps predictions and labels have differing shapes", preds.shape, labels.shape)

        visibility = regression_labels[..., 2]

        # Calculate squared errors between each pixel
        squared_errors = (preds - heatmap_labels) ** 2

        # Mask out the non visible heatmaps
        mask = (visibility != -1).float()

        # Expand mask to same shape as heatmap
        mask = mask[:, :, None, None]
        mask = mask.expand_as(squared_errors)

        masked_squared_errors = squared_errors * mask

        loss = masked_squared_errors.sum() / (mask.sum() + 1e-8)

        return loss



if __name__ == "__main__":
    # Batch of 2 images, each with 4 heatmaps (keypoint)
    outputs = torch.tensor([[[[0.2218, 0.4659, 0.5017],
          [0.2247, 0.6974, 0.6436],
          [0.1161, 0.1973, 0.0249]],

         [[0.1417, 0.4200, 0.5672],
          [0.2178, 0.9498, 0.8920],
          [0.5144, 0.7793, 0.0494]],

         [[0.0429, 0.6788, 0.4906],
          [0.9680, 0.8928, 0.4555],
          [0.5777, 0.7187, 0.1422]],

         [[0.1695, 0.4089, 0.1025],
          [0.6283, 0.8367, 0.2132],
          [0.4074, 0.6184, 0.6538]]],


        [[[0.7250, 0.2910, 0.5027],
          [0.1152, 0.2841, 0.7886],
          [0.7914, 0.7484, 0.1203]],

         [[0.6971, 0.8450, 0.6313],
          [0.8263, 0.7762, 0.8329],
          [0.0753, 0.6956, 0.9221]],

         [[0.6996, 0.7944, 0.7386],
          [0.4804, 0.1543, 0.7622],
          [0.8840, 0.8845, 0.6304]],

         [[0.9873, 0.6411, 0.6023],
          [0.2662, 0.3910, 0.4571],
          [0.6577, 0.4469, 0.7536]]]])

    labels = torch.tensor([[[[1.0000, 0.1353, 0.0003],
          [0.1353, 0.0183, 0.0000],
          [0.0003, 0.0000, 0.0000]],

         [[0.0183, 0.1353, 0.0183],
          [0.1353, 1.0000, 0.1353],
          [0.0183, 0.1353, 0.0183]],

         [[0.0000, 0.0003, 0.0000],
          [0.0003, 0.0183, 0.1353],
          [0.0000, 0.1353, 1.0000]],

         [[0.0000, 0.0003, 1.0000],
          [0.0000, 0.0183, 0.1353],
          [0.0000, 0.0000, 0.0003]]],


        [[[1.0000, 0.1353, 0.0003],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.1353, 1.0000, 0.1353],
          [0.0183, 0.1353, 0.0183],
          [0.0000, 0.0003, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0183, 0.1353, 1.0000],
          [0.0000, 0.0003, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0003, 0.1353, 1.0000],
          [0.0000, 0.0183, 0.1353]]]])


    regression_labels = torch.tensor([
        [[10, 20, 0], [29, 41, -1], [49, 59, 0], [70, 79, 0]],
        [[12, 22, 1], [29, 39, 1], [52, 62, -1], [70, 82, 0]]
    ])

    criterion = HeatmapLoss()

    loss = criterion(outputs, labels, regression_labels)
    print("Total loss:", loss)




