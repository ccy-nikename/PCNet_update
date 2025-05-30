# -*- coding: utf-8 -*-
import argparse
import json
import logging
import numpy as np
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', back_count=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=back_count,
                                               encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(th)


def restore_vessel(data):
    r, b = np.zeros_like(data), np.zeros_like(data)

    r[data == 2] = 1
    b[data == 1] = 1

    res = np.zeros((*data.shape, 3))

    res[:, :, 0] = r
    res[:, :, 2] = b

    return res


def adjust_lr(optimize, decay_rate):
    for param_group in optimize.param_groups:
        param_group['lr'] *= decay_rate
    print('The learning rate has changed')


def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def montage_for_trans(out_list, size):
    w, h = size
    s = out_list[0].shape[1]

    m = w // s + (0 if w % s == 0 else 1)
    n = h // s + (0 if h % s == 0 else 1)

    tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

    prediction = np.zeros((n * s, m * s))

    for j in range(m):
        for i in range(n):
            prediction[i * s: (i + 1) * s, j * s: (j + 1) * s] = out_list[j * n + i]

    prediction = prediction[tmp_h: tmp_h + h, tmp_w: tmp_w + w]

    return prediction


def montage_for_circle(out_list, size):
    # print(len(out_list[0]))
    w, h = size
    s = out_list[0][0].shape[0]
    res = []

    m = w // s + (0 if w % s == 0 else 1)
    n = h // s + (0 if h % s == 0 else 1)

    tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

    for item in range(len(out_list[0])):
        prediction = np.zeros((n * s, m * s))
        for j in range(m):
            for i in range(n):
                prediction[i * s: (i + 1) * s, j * s: (j + 1) * s] = out_list[j * n + i][item]

        prediction = prediction[tmp_h: tmp_h + h, tmp_w: tmp_w + w]
        res.append(prediction)

    return res


class MeanList(object):
    def __init__(self):
        self.data = []

    def append(self, x):
        self.data.append(x)

    def mean(self):
        return np.round(np.mean(self.data), 4)


def montage_one_hot(out_av_list, size):
    w, h = size
    s = out_av_list[0].shape[1]

    m = w // s + (0 if w % s == 0 else 1)
    n = h // s + (0 if h % s == 0 else 1)

    tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

    pre = np.zeros((n * s, m * s, 3))

    for j in range(m):
        for i in range(n):
            pre[i * s: (i + 1) * s, j * s: (j + 1) * s, 0] = out_av_list[j * n + i][0, ...]
            pre[i * s: (i + 1) * s, j * s: (j + 1) * s, 2] = out_av_list[j * n + i][2, ...]

    pre = pre[tmp_h: tmp_h + h, tmp_w: tmp_w + w, :]

    return pre


class DotDict(dict):
    """A dictionary that supports dot notation."""

    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value


def json_type(s):
    try:
        return json.load(open(s, 'r'))
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON")
