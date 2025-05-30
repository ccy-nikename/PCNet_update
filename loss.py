# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class SegmentationLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super(SegmentationLoss, self).__init__()
        self.ep = ep

        pos_weight = torch.FloatTensor([5.0]).to(config.device)
        self.final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(config.device)

        self.focal_loss = FocalLoss().to(config.device)
        self.criterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = (predict * label).sum()
        union = predict.sum() + label.sum()
        dice = (2. * intersection + self.ep) / (union + self.ep)
        return 1 - dice

    def forward(self, predict_list, predict_out, label):
        loss_multi = 0

        for idx, predict in enumerate(predict_list):
            loss_multi += (1 / (len(predict_list) - idx)) * self.criterion(predict, label)
            loss_multi += (1 / (len(predict_list) - idx)) * self.dice_loss(predict, label)

        loss_last = self.final_criterion(predict_out, label)
        loss_last += self.dice_loss(torch.sigmoid(predict_out), label)

        return loss_multi + loss_last


class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma
        self.criterion = nn.BCELoss()

    def forward(self, inputs, targets):
        bce_loss = self.criterion(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        f_loss = at * (1 - pt) ** self.gamma * bce_loss
        return f_loss.mean()


class CircleLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.BCELoss().to(config.device)
        self.ep = 1e-6

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict_list, label):
        loss = 0

        for idx, predict in enumerate(predict_list):
            loss += ((1 + idx) / len(predict_list)) * self.criterion(predict, label.float())
            loss += ((1 + idx) / len(predict_list)) * self.dice_loss(predict, label.float())

        return loss


class SegmentationLossAdjust(nn.Module):
    def __init__(self, config, ep=1e-6):
        super(SegmentationLossAdjust, self).__init__()
        self.ep = ep
        self.criterion = nn.BCELoss().to(config.device)
        self.focal_loss = FocalLoss().to(config.device)
        self.final_criterion = nn.BCEWithLogitsLoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict_list, predict_out, label, flag=True):
        loss_multi = 0

        if flag:

            for idx, predict in enumerate(predict_list):
                loss_multi += (1 / (len(predict_list) - idx)) * self.criterion(predict, label)
                loss_multi += (1 / (len(predict_list) - idx)) * self.dice_loss(predict, label)

        else:

            loss_multi += self.final_criterion(predict_out, label)
            loss_multi += self.dice_loss(torch.sigmoid(predict_out), label)

        return loss_multi


class ArteryVeinLoss(nn.Module):

    def __init__(self, config, beta=(5, 1, 4), ep=1e-6):
        super().__init__()

        pos_weight = torch.FloatTensor([5.0]).to(config.device)

        self.artery_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(config.device)
        self.vein_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(config.device)
        self.uncertain_criterion = nn.BCEWithLogitsLoss().to(config.device)

        self.beta = beta
        self.ep = ep

    def dice_loss(self, prediction, label):
        intersection = 2 * torch.sum(prediction * label) + self.ep
        union = torch.sum(prediction) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, prediction, label):
        loss = 0

        for i, criterion in enumerate([self.artery_criterion, self.uncertain_criterion, self.vein_criterion]):
            p, t = prediction[:, i, ...], label[:, i, ...]
            loss += self.beta[i] * (criterion(p, t) + 0.5 * self.dice_loss(torch.sigmoid(p), t))

        return loss / 10


class VesselLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()
        self.ep = ep
        self.criterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict_list, label):
        loss_multi = 0

        for idx, predict in enumerate(predict_list):
            loss_multi += (1 / (len(predict_list) - idx)) * self.criterion(predict, label)

        return loss_multi


class TransformLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()
        self.ep = ep
        self.criterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict, label):
        loss_multi = 0

        loss_multi += self.criterion(predict, label)
        loss_multi += self.dice_loss(predict, label)

        return loss_multi


class SerialLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()
        self.ep = ep
        self.criterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict, label):
        loss_multi = 0

        for p in predict[:-1]:
            loss_multi += self.criterion(p, label)
            loss_multi += 0.1 * self.dice_loss(p, label)

        loss_multi /= len(predict)

        loss_multi += self.criterion(predict[-1], label)
        loss_multi += 0.1 * self.dice_loss(predict[-1], label)

        return loss_multi


class IGLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()
        self.ep = ep
        self.criterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def tversky_loss(self, predict, label):

        flat_label = label.flatten()
        flat_prediction = predict.flatten()
        tp = torch.sum(flat_label * flat_prediction)
        fn = torch.sum(flat_label * (1 - flat_prediction))
        fp = torch.sum((1 - flat_label) * flat_prediction)

        tversky = (tp + self.ep) / (tp + 0.3 * fp + 0.7 * fn + self.ep)

        return 1 - tversky

    def forward(self, predict, label):
        loss_multi = 0

        for idx, p in enumerate(predict):
            loss_multi += (1 / (len(predict) - idx)) * (self.criterion(p, label) + 0.1 * self.dice_loss(p, label))
        return loss_multi


class MultiLoss(nn.Module):
    def __init__(self, config, ep=1e-6, alpha=0.3):
        super().__init__()
        self.ep = ep
        self.alpha = alpha
        self.ig_criterion = IGLoss(config)
        self.pi_criterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def tversky_loss(self, predict, label):

        flat_label = label.flatten()
        flat_prediction = predict.flatten()
        tp = torch.sum(flat_label * flat_prediction)
        fn = torch.sum(flat_label * (1 - flat_prediction))
        fp = torch.sum((1 - flat_label) * flat_prediction)

        tversky = (tp + self.ep) / (tp + 0.3 * fp + 0.7 * fn + self.ep)

        return 1 - tversky

    def forward(self, predict_list, prediction, label):
        loss = 0
        loss += self.alpha * self.ig_criterion(predict_list, label)
        loss += (1 - self.alpha) * (self.pi_criterion(prediction, label) + 0.5 * self.dice_loss(prediction, label))

        return loss


class U2NetLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()
        self.ep = ep
        self.criterion = nn.BCEWithLogitsLoss().to(config.device)

    def forward(self, prediction, label):
        loss = 0
        for p in prediction:
            loss += self.criterion(p, label)
        return loss


class IGBUSLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super().__init__()

        self.ep = ep
        self.criterion = nn.BCEWithLogitsLoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict, label):
        loss_multi = 0

        for idx, p in enumerate(predict):
            loss_multi += (1 / (len(predict) - idx)) * (
                    self.criterion(p, label) + 0.1 * self.dice_loss(torch.sigmoid(p), label))
        return loss_multi


class MultiBUSLoss(nn.Module):
    def __init__(self, config, ep=1e-6, alpha=0.4):
        super().__init__()
        self.ep = ep
        self.alpha = alpha
        self.ig_criterion = IGBUSLoss(config)
        self.pi_criterion = nn.BCEWithLogitsLoss().to(config.device)

    def dice_loss(self, predict, label):
        intersection = 2 * torch.sum(predict * label) + self.ep
        union = torch.sum(predict) + torch.sum(label) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predict_list, prediction, label):
        loss = 0
        loss += self.alpha * self.ig_criterion(predict_list, label)
        loss += (1 - self.alpha) * (
                self.pi_criterion(prediction, label) + 0.1 * self.dice_loss(torch.sigmoid(prediction), label))

        return loss


class SerialConnectedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss().to(config.device)

    def forward(self, predict_list, label):
        loss_multi = 0
        for idx, p in enumerate(predict_list):
            if idx != len(predict_list) - 1:
                loss_multi += (1 / len(predict_list)) * self.criterion(p, label)
            else:
                loss_multi += self.criterion(p, label)

        return loss_multi


class MATNetLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ep = 1e-6
        self.criterion = nn.BCELoss().to(config.device)

    def forward(self, predict_list, label):
        loss = 0
        for p in predict_list:
            loss += self.criterion(p, label)
        return loss


class SwinPALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(config.device)
        self.avg = nn.AvgPool2d(kernel_size=31, stride=1, padding=15).to(config.device)

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(self.avg(mask) - mask)
        wbce = self.criterion(pred, mask)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()
    
    def forward(self, pred, mask):
        return self.structure_loss(pred, mask)