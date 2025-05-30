# -*- coding: utf-8 -*-
import copy
import os
import shutil
import warnings
import cv2
import random

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.util import random_noise
import skimage

from torch.nn.functional import grid_sample
from torchvision.transforms import ToTensor, ToPILImage

warnings.simplefilter('ignore')


class Process(object):
    def __init__(self, config):
        self.data_path = config.data_path
        self.split_num = config.split_num
        self.sources = config.data_sources
        self.data_set = config.data_sets

        os.makedirs(self.data_path, exist_ok=True)

    def __adjust(self):
        size = (512, 512)

        for k in self.certain_data.keys():
            self.certain_data[k] = self.certain_data[k].resize(size)
            if k == 'label' or k == 'vessel':
                self.certain_data[k] = Image.fromarray(np.uint8(np.round(np.array(self.certain_data[k]) / 255) * 255))

    def get_name(self):

        sign = self.data_set

        self.train_img_name = [i for i in os.listdir(os.path.join(self.sources[sign], 'training', 'images')) if i.endswith('.png')]
        self.test_img_name = [i for i in os.listdir(os.path.join(self.sources[sign], 'test', 'images')) if i.endswith('.png')]
        self.val_img_name = self.test_img_name

    def run_train(self):
        self.__del()
        sign = self.data_set
        self.get_name()

        with tqdm(total=len(self.train_img_name) + len(self.val_img_name), desc=f'{sign}', ncols=60) as bar:

            for name in self.train_img_name:
                image = Image.open(os.path.join(self.sources[sign], 'training', 'images', name))
                label = Image.open(os.path.join(self.sources[sign], 'training', 'label', name)).convert('RGB')
                try:
                    vessel = Image.open(os.path.join(self.sources[sign], 'training', 'vessel', name)).convert('L')
                except:
                    vessel = Image.open(os.path.join(self.sources[sign], 'training', 'label', name)).convert('L')

                self.certain_name = name

                self.certain_data = {'image': image, 'label': label, 'vessel': vessel}
                self.__adjust()
                self.__produce_training_image()
                bar.update(1)

            for idx, name in enumerate(self.val_img_name):

                image = Image.open(os.path.join(self.sources[sign], 'test', 'images', name))
                label = Image.open(os.path.join(self.sources[sign], 'test', 'label', name)).convert('RGB')
                try:
                    vessel = Image.open(os.path.join(self.sources[sign], 'test', 'vessel', name)).convert('L')
                except:
                    vessel = Image.open(os.path.join(self.sources[sign], 'test', 'label', name)).convert('L')

                self.certain_name = sign + '_' + name

                self.certain_data = {'image': image}

                self.__adjust()

                self.certain_data['label'] = label
                self.certain_data['vessel'] = vessel

                for k in self.certain_data.keys():
                    img = self.certain_data[k]
                    img.save(os.path.join(self.data_path, 'validation', k, self.certain_name))
                    img.save(os.path.join(self.data_path, 'test', k, self.certain_name))

                bar.update(1)

    def run_test(self):

        self.get_name()

        sign = self.data_set

        with tqdm(total=len(self.test_img_name), desc=f'{sign}', ncols=60) as bar:

            for idx, name in enumerate(self.test_img_name):

                image = Image.open(os.path.join(self.sources[sign], 'test', 'images', name))
                label = Image.open(os.path.join(self.sources[sign], 'test', 'label', name))

                self.certain_name = sign + '_' + name

                image = np.array(image)

                self.certain_data = {'image': image}

                self.__adjust()

                self.certain_data['label'] = label

                for k in self.certain_data.keys():
                    img = self.certain_data[k]
                    img.save(os.path.join(self.data_path, 'test', k, self.certain_name))

                bar.update(1)

    def __del(self):

        for item in ['train', 'test', 'validation']:
            if os.path.exists(os.path.join(self.data_path, item)):
                shutil.rmtree(os.path.join(self.data_path, item))

            _dir = ['image', 'label', 'vessel']
            for name in _dir:
                os.makedirs(os.path.join(self.data_path, item, name), exist_ok=True)

    def __produce_training_image(self):

        for index in range(self.split_num):

            data = copy.deepcopy(self.certain_data)

            data = warp(noise(rotate(data)))
            # data = warp(rotate(data))

            for k in data.keys():
                tmp = data[k]
                if k == 'label' or k == 'vessel':
                    tmp = np.round(np.array(tmp) / 255) * 255
                    tmp = Image.fromarray(np.uint8(tmp))
                tmp.save(
                    os.path.join(self.data_path, 'train', k, self.certain_name.split('.')[0] + f'_{index}' + '.png'))


def rotate(data):
    if random.random() < 0.2:
        return data

    res = dict()
    angle = random.choice(range(90))
    mid = [0 if random.random() < 0.5 else 1, 0 if random.random() < 0.5 else 1]

    for k in data.keys():
        tmp = cv2.cvtColor(np.asarray(data[k]), cv2.COLOR_RGB2BGR)
        if mid[0]:
            tmp = cv2.flip(tmp, 1)
        if mid[1]:
            tmp = cv2.flip(tmp, 0)

        tmp = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        tmp = tmp.rotate(angle)

        if k == 'label' or k == 'vessel':
            tmp = np.round(np.array(tmp) / 255) * 255
            tmp = Image.fromarray(np.uint8(tmp))

        res[k] = tmp
    return res


def noise(data):
    if random.random() < 0.8:
        return data

    res = dict()
    var = np.random.choice([0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002])
    for k in data.keys():
        tmp = data[k]
        if k == 'image':
            tmp = random_noise(np.array(tmp), var=var)
            tmp = Image.fromarray(skimage.util.dtype.img_as_ubyte(tmp))
        res[k] = tmp
    return res


def warp(data):
    if random.random() < 0.5:
        return data

    res = dict()
    for k in data.keys():
        img = data[k]
        img = ToTensor()(img).unsqueeze(0)
        _, _, h, w = img.shape
        xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
        xx = xx.view(1, h, w)
        yy = yy.view(1, h, w)
        grid = torch.cat((xx, yy), 0).unsqueeze(0).float()
        grid = 2.0 * grid / (h - 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        img = grid_sample(img, grid)
        img = ToPILImage()(img.squeeze())

        if k == 'label' or k == 'vessel':
            img = np.round(np.array(img) / 255) * 255
            img = Image.fromarray(np.uint8(img))

        res[k] = img
    return res
