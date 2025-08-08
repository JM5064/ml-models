"""
MPII Human Pose dataset with heatmaps, x, y offset maps, and keypoints
"""

import os
import numpy as np
import scipy.io
import jsonyx as json

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MPIIDataset(Dataset):

    def __init__(self, images_dir, annotations_json):
        self.dataset = json.load(open(annotations_json, "r"))
        self.images_dir = images_dir


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        raise NotImplementedError()