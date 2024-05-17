# Dataset class for database managment

import cv2

from PIL import Image
from torch.utils.data import Dataset
from matplotlib import image as img

import numpy as np

# Opt_dir is responsible for correct loading images
# Set this for your database location

class EmotionsDataset(Dataset):
    def __init__(self, db, opt_dir):
        super().__init__()
        self.db = db
        self.opt_dir = opt_dir
        self.horz_max = 0
        self.vert_max = 0

        # Searching for largest image
        for idx in self.db.index:
            horz = abs(int(self.db['box_left'][idx]) - int(self.db['box_right'][idx]))
            vert = abs(int(self.db['box_bottom'][idx]) - int(self.db['box_top'][idx]))
            if horz > self.horz_max:
                self.horz_max = horz
            if vert > self.vert_max:
                self.vert_max = vert

    def __len__(self):
        return self.db.shape[0]

    def __getitem__(self, index):
        pict = img.imread(self.opt_dir + self.db['image'][index])
        label = self.db['label'][index]

        return {'image': pict, 'label': label}
    