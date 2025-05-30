# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       : 2024/5/15 20:49
# Author     : Chen Chouyu
# Email      : chenchouyu2020@gmail.com
"""
import argparse

from torch.utils.data import DataLoader
from loader import BinaryDataset
from process import Process
from test import test

from utils import json_type, DotDict

import Net

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--model_name', default='PCNet')
    arg.add_argument('--alpha', type=float, default=0.5)
    arg.add_argument('--depth', type=int, default=5)
    arg.add_argument('--progressive_mode', default='dual')
    arg.add_argument('--config', type=json_type, default='./config_DRIVE.json')
    arg.add_argument('--module1', action='store_false')
    arg.add_argument('--module2', action='store_false')
    arg = arg.parse_args()

    config = DotDict(arg.config)

    if config.preprocess:
        pre = Process(config.preprocess_setting)
        pre.run_train()

    if arg.model_name == 'PCNet':
        alpha = arg.alpha
        model = Net.__dict__['PCNet'].__dict__[arg.model_name](
            in_channel=1, out_channel=1, depth=arg.depth, alpha=alpha, progressive_mode=arg.progressive_mode,
            module1=arg.module1, module2=arg.module2
        )
    else:
        raise 'Model Error'

    # print(model)
    from train import train, warming


    data = BinaryDataset(config.dataloader_setting, mode='train')
    train_loader = DataLoader(data, shuffle=True, batch_size=config.train_setting.batch_size)
    val_loader = DataLoader(BinaryDataset(config.dataloader_setting, mode='validation'))
    model_name = arg.model_name + '_' + arg.progressive_mode + '_' + config.preprocess_setting.data_sets

    warming_model = model.module1
    warming(config.train_setting, warming_model, model_name,
            train_loader=train_loader, val_loader=val_loader)

    train(config.train_setting, model, model_name, train_loader=train_loader, val_loader=val_loader)
