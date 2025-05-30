# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
from collections import defaultdict


class BinaryDataset(Dataset):
    def __init__(self, config, mode):
        super().__init__()

        path = os.path.join(config.path, mode)

        self.mode = mode

        self.img_path = os.path.join(path, 'image')
        self.lab_path = os.path.join(path, 'vessel')

        self.itemList = os.listdir(self.img_path)

        self.transforms = transforms.Compose([
            ToTensor()
        ])
        self.transToTensor = ToTensor()

    def __len__(self):
        return len(self.itemList)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.img_path, self.itemList[item])).convert('L')
        lab = Image.open(os.path.join(self.lab_path, self.itemList[item])).convert('L')

        original_size = lab.size
        current_size = img.size

        img = self.transforms(img)
        lab = self.transToTensor(lab)

        return img, lab, original_size, current_size, self.itemList[item]


class ExtractGreenChannel(object):
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
            green_channel = img[:, :, 1]
            return Image.fromarray(green_channel)
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.size(0) == 3:
                green_channel = img[1, :, :]
                return green_channel.unsqueeze(0)
            else:
                raise ValueError("Input tensor is not a valid 3-channel image")
        else:
            raise TypeError("Input should be a PIL Image or a Tensor")


class EmployCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img = img.squeeze()
        clahe_img = clahe.apply(img)
        return clahe_img
