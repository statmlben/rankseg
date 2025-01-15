# Author: Ben Dai <bendai@cuhk.edu.hk>
# https://github.com/statmlben/rankseg

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob

ignore_label = 255
ID_TO_TRAINID={}
# ID_TO_TRAINID = {0: ignore_label, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6:5, 7:6, 8:7, 
#                 9:8, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18, 20:19}


class kvasirSEGDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        # self.palette = palette.ADE20K_palette
        super(kvasirSEGDataset, self).__init__(**kwargs)

    def _set_files(self):
        # if self.split in  ["training", "validation"]:
        #     self.image_dir = os.path.join(self.root, 'images', self.split)
        #     self.label_dir = os.path.join(self.root, 'masks', self.split)
        #     self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        # else: raise ValueError(f"Invalid split name {self.split}")
        if self.split in  ["train", "val"]:
            self.image_dir = os.path.join(self.root, 'images_split', self.split, 'jpg_images')
            self.label_dir = os.path.join(self.root, 'masks_split', self.split, 'jpg_masks')
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.jpg')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path).convert('1'), dtype=np.int32) 
        return image, label, image_id


class kvasirSEG(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = kvasirSEGDataset(**kwargs)
        super(kvasirSEG, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
