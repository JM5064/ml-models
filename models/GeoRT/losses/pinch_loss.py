from typing import Any
import torch
import torch.nn as nn


class PinchLoss(nn.Module):

    def __init__(self, threshold):
        super().__init__()

        # Pinching distance threshold
        self.threshold = threshold


    def forward(self, pred_positions, labels):
        # Create mask for GT fingers which are pinching

        # For the fingers which are supposed to be pinching, 
        # calculate loss based on the distance between those two fingers in pred_pos


        return


