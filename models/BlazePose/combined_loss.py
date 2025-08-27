import numpy as np

import torch
import torch.nn as nn

from .regression_loss import RegressionLoss
from .heatmap_loss import HeatmapLoss


class CombinedLoss(nn.Module):

    def __init__(self, alpha=1.5):
        super().__init__()

        self.regression_loss_func = RegressionLoss()
        self.heatmap_loss_func = HeatmapLoss()

        self.alpha = alpha


    def forward(self, regression_preds, regression_labels, heatmap_preds, heatmap_labels, offset_masks):
        regression_loss = self.regression_loss_func(regression_preds, regression_labels)
        heatmap_loss = self.heatmap_loss_func(heatmap_preds, heatmap_labels, offset_masks, regression_labels)

        combined_loss = regression_loss + self.alpha * heatmap_loss

        return combined_loss
