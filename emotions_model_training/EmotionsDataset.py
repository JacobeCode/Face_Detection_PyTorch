# Dataset class for database managment

import cv2

from PIL import Image
from torch.utils.data import Dataset

import numpy as np

class EmotionsDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass
    def __getitem__(self, index):
        return super().__getitem__(index)