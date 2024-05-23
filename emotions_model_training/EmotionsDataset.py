# Dataset class for database managment

import cv2
import torch

from PIL import Image
from torch.utils.data import Dataset
from matplotlib import image as img
from torchvision.transforms import v2
from torchvision.transforms.functional import resize
from add_files.torch_transforms import torch_transforms

import numpy as np
import torchvision.transforms as ttinter

# Opt_dir is responsible for correct loading images
# Set this for your database location

class EmotionsDataset(Dataset):
    def __init__(self, db, opt_dir, horz_max=0, vert_max=0, padding=False):
        super().__init__()
        self.db = db
        self.opt_dir = opt_dir
        self.horz_max = horz_max
        self.vert_max = vert_max
        self.padding = padding
        self.transforms = v2.Compose([
            v2.Resize([256, 256], ttinter.InterpolationMode.BICUBIC, max_size=256, antialias=True)
        ])

        # Searching for largest image (if to pass this step if size provided)
        if self.horz_max == 0 or self.vert_max == 0:
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
        # Getting proper image from database and box for faces
        pict = Image.open(self.opt_dir + self.db['image'][index])
        box = (int(self.db['box_left'][index]), int(self.db['box_top'][index]), int(self.db['box_right'][index]), int(self.db['box_bottom'][index]))
        
        # Preparing "only face" image by cropping and getting labels
        crop_pict = np.float32(pict.crop(box))
        label = np.int64(self.db['label'][index])

        # Redesigning order of dimension for training
        crop_pict = np.moveaxis(crop_pict, 2, 0)

        if self.padding is True:
            img_processed = self.padding(crop_pict)        
        else:
            # Other transforms - resize
            img_processed = resize(torch.tensor(crop_pict), [256, 256], ttinter.InterpolationMode.BICUBIC, antialias=True)

        return {'image': img_processed, 'label': label}
    

    # Function for padding the image (manual)
    def padding(self, pict):
        # Chaning dimension for training in np.float32
        crop_pict = np.float32(pict)

        # Manual padding to proper maximal value
        # Getting proper values for padding
        pad_vert = (self.vert_max - crop_pict.shape[1]) / 2
        pad_horz = (self.horz_max - crop_pict.shape[2]) / 2

        # Padding sides (with covering half values - more in: detect_analysis.py)
        if pad_horz % 2 == 0 or pad_horz % 2 == 1:
            pad_right = int(pad_horz)
            pad_left = int(pad_horz)
        else:
            pad_right = int(pad_horz) + 1
            pad_left = int(pad_horz)

        if pad_vert % 2 == 0 or pad_horz % 2 == 1:
            pad_bottom = int(pad_vert)
            pad_top = int(pad_vert)
        else:
            pad_bottom = int(pad_vert) + 1
            pad_top = int(pad_vert)
            
        pads = ((pad_left, pad_right), (pad_top, pad_bottom))

        # Preferable for torch - np.float32
        img_processed = np.ndarray((3, self.horz_max, self.vert_max), np.float32)

        # Padding image with constant median value
        for i, x in enumerate(crop_pict):
            cons = int(np.median(x))
            x_p = np.pad(x, pads, 'constant', constant_values=cons)
            img_processed[i,:,:] = x_p

        return img_processed
    