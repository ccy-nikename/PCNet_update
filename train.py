# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from loss import IGLoss, MultiLoss
from utils import count_parameters, MeanList

from sklearn.metrics import accuracy_score, f1_score, recall_score


def warming(config, model, model_name, train_loader, val_loader):
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = IGLoss(config).to(config.device)
    model_save_warming_path = os.path.join(config.save_path, model_name, 'warming')

    os.makedirs(model_save_warming_path, exist_ok=True)

    warming_best_f1 = 0

    print('Generate network scale: ' + str(np.round(count_parameters(model) / 1e6, 3)) + 'M')

    print('Start Warming ......')
    best_epoch = 0

    for epoch in range(config.warming_epoch):
        with tqdm(total=len(train_loader), ncols=80, desc=f'[0/{len(train_loader)}] loss = 0') as bar:
            model.train()
            for i, data in enumerate(train_loader):
                img, lab, _, _, _ = data
                img = img.to(config.device)
                lab = lab.to(config.device)

                prediction_list = model(img)
                loss = criterion(prediction_list, lab)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.set_description(f'[{i + 1}/{len(train_loader)}] loss = {np.round(loss.item(), 4)}')
                bar.update(1)
            model.eval()

            f1_avg = MeanList()

            for i, data in enumerate(val_loader):

                img, lab, original_size, current_size, name = data
                lab = lab.numpy().squeeze()
                lab = np.where(lab > 0.5, 1, 0)

                original_size = tuple(map(int, original_size))

                with torch.no_grad():
                    img = img.to(config.device)
                    predict_list = model(img)
                    predicted = [i.cpu().numpy().squeeze() for i in predict_list]

                predicted = [cv2.resize(m, original_size) for m in predicted]
                compute_prediction = [np.where(m > 0.5, 1, 0) for m in predicted]
                f1 = [f1_score(m.flatten(), lab.flatten()) for m in compute_prediction]

                f1_avg.append(f1)

            f1 = f1_avg.mean()
            print(f'Certain epoch {epoch + 1} F1 = {f1}')

            if warming_best_f1 < f1:
                path_checkpoint = os.path.join(model_save_warming_path, f'warming_model.pth')
                torch.save(model.state_dict(), path_checkpoint)
                warming_best_f1 = f1
                best_epoch = epoch

    print(f'Best f1 = {warming_best_f1} in epoch {best_epoch + 1}')
    print('Warming Completed ...')


def adjust_learning_rate(optimizer, epoch, initial_lr, step_size=80, gamma=0.9):
    lr = initial_lr * ((1 - epoch / step_size) ** gamma)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(config, model, model_name, train_loader, val_loader):
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    criterion = MultiLoss(config).to(config.device)

    validate_save_path = os.path.join(config.save_path, model_name, 'validation')
    model_save_path = os.path.join(config.save_path, model_name, 'model')
    middle_save_path = os.path.join(config.save_path, model_name, 'middle')

    model_save_warming_path = os.path.join(config.save_path, model_name, 'warming')

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(validate_save_path, exist_ok=True)

    training_best_acc = 0.96
    training_best_f1 = 0.8

    print('Network scale: ' + str(np.round(count_parameters(model) / 1e6, 3)) + 'M')
    print('Start Training ...')

    new_state_dict = {}
    pretrained_state_dict = torch.load(os.path.join(model_save_warming_path, 'warming_model.pth'))
    for key, value in pretrained_state_dict.items():
        new_key = "module1." + key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)

    for epoch in range(config.epoch):

        show_loss = 0
        model.train()

        ttt = adjust_learning_rate(optimizer, epoch, 0.002)
        print(f'Now lr: {ttt}')

        with tqdm(total=len(train_loader), ncols=80, desc=f'[{epoch + 1}/{config.epoch}] Loss=inf') as bar:
            for i, data in enumerate(train_loader):
                img, lab, _, _, _ = data
                if torch.cuda.is_available():
                    img = img.to(config.device)
                    lab = lab.to(config.device)

                prediction_list, prediction, _ = model(img)
                loss = criterion(prediction_list, prediction, lab)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.set_description(f'[{epoch + 1}/{config.epoch}] Loss={np.round(loss.item(), 4)}')
                show_loss += loss.item()

                bar.update(1)

        print('Segmentation Loss: %.4f' % (show_loss / len(train_loader)))

        model.eval()
        os.makedirs(os.path.join(validate_save_path, str(epoch + 1)), exist_ok=True)
        os.makedirs(os.path.join(middle_save_path, str(epoch + 1)), exist_ok=True)

        acc_avg, f1_avg, rec_avg = MeanList(), MeanList(), MeanList()

        for i, data in enumerate(val_loader):
            img, lab, original_size, current_size, name = data

            lab = lab.numpy().squeeze()

            original_size = tuple(map(int, original_size))

            with torch.no_grad():
                img = img.to(config.device)
                _, prediction, _ = model(img)
                prediction = prediction.cpu().numpy().squeeze()
            save_prediction = cv2.resize(prediction, original_size)
            cv2.imwrite(
                os.path.join(validate_save_path, str(epoch + 1), name[0]),(save_prediction * 255).astype(np.uint8)
            )

            compute_prediction = np.where(save_prediction > 0.5, 1, 0)
            lab = np.where(lab > 0.5, 1, 0)

            acc = accuracy_score(compute_prediction.flatten(), lab.flatten())
            f1 = f1_score(compute_prediction.flatten(), lab.flatten())
            rec = recall_score(compute_prediction.flatten(), lab.flatten())

            acc_avg.append(acc)
            f1_avg.append(f1)
            rec_avg.append(rec)

        acc, f1, rec = acc_avg.mean(), f1_avg.mean(), rec_avg.mean()

        if training_best_f1 < f1 or training_best_acc < acc:

            path_checkpoint = os.path.join(model_save_path, 'epoch_{}_{}.pth'.format(epoch + 1, acc))
            torch.save(model.state_dict(), path_checkpoint)

            training_best_f1 = f1
            training_best_acc = acc

            print('save, f1: %.4f, acc: %.4f, rec: %.4f' % (f1, acc, rec))
        else:
            print('f1: %.4f, acc: %.4f, rec: %.4f' % (f1, acc, rec))

    print('Training Completed ...')
