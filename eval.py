import os

from metrics.ms_ssim import MS_SSIM, SSIM
import torch
from torch import nn
import cv2

channels = 1

target_path = './scan/'
pred_path = './results/'
ms_ssim = SSIM(channel=channels)
scores = []
for pred_name in os.listdir(pred_path):
    target_name = pred_name.split('_')[0] + '.png'
    target = cv2.imread(target_path + target_name)
    pred = cv2.imread(pred_path + pred_name)
    target = cv2.resize(target, (pred.shape[1], pred.shape[0]))
    temp = target.copy()
    if channels == 1:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        if pred_name.startswith('44'):
            cv2.imwrite(target_name, target)
        target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        if pred_name.startswith('44'):
            cv2.imwrite(pred_name, pred)
        pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    else:
        target = torch.from_numpy(target).unsqueeze(0).float()
        target = target.permute(0, 3, 1, 2)
        pred = torch.from_numpy(pred).unsqueeze(0).float()
        pred = pred.permute(0, 3, 1, 2)
    score = ms_ssim(pred, target)
    scores.append(score)

