import torch
from torch.utils import data
from torch.autograd import Variable, Function
import numpy as np
import sys, os, math
import cv2
import time
import re
import random
from scipy.interpolate import griddata
from torch.utils.data.distributed import DistributedSampler



class AverageMeter(object):
    def __init__(self, loss_dict):
        self.Loss = {k:0 for k in loss_dict.keys()}
        self.Loss['total'] = 0
        self.cnt = 0

    def reset(self):
        self.Loss = {k:0 for k in self.Loss.keys()}
        self.cnt = 0

    def update(self, val, loss_name, num):
        self.Loss[loss_name] += val
        self.Loss['total'] += val
        self.cnt += num


class Trainer(object):

    def __init__(self, args, model, output_path, start_epoch, optimizer, loss_dict,
                 dataset, data_path, data_path_validate, data_preproccess=True, validate=False):
        # base
        self.args = args
        self.output_path = output_path
        self.model = model
        self.epoch = start_epoch
        self.iteration = 0
        self.validate = validate

        # dataset
        self.dataset = dataset
        self.data_path = data_path
        self.data_path_validate = data_path_validate
        self.data_preproccess = data_preproccess
        self.train_loader = self.loadData('train')
        if args.rank == 0:
            print("{} images have been loaded in training dataset".format(len(self.train_loader.dataset)))
        self.val_loader = self.loadData('val')
        if args.rank == 0:
            print("{} images have been loaded in validation dataset".format(len(self.val_loader.dataset)))

        # loss
        total_iter = len(self.train_loader) * args.n_epoch
        self.optimizer = optimizer
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iter, eta_min=1e-6, last_epoch=-1)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.25*args.n_epoch), int(0.5*args.n_epoch), int(0.75*args.n_epoch)], gamma=0.5)
        self.loss_dict = loss_dict
        self.loss_log = AverageMeter(loss_dict)
        self.val_loss = AverageMeter(loss_dict)


    def loadData(self, data_split):
        if data_split == 'train':
            data_set = self.dataset(self.data_path, split=data_split)
            if self.args.distributed:
                data_loader = data.DataLoader(data_set, batch_size=self.args.batch_size, num_workers=8, drop_last=True, pin_memory=True,
                                        sampler=DistributedSampler(data_set))
            else:
                data_loader = data.DataLoader(data_set, batch_size=self.args.batch_size, num_workers=8, drop_last=True, pin_memory=True,
                                              shuffle=True)
        elif data_split == 'val':
            data_set = self.dataset(self.data_path_validate, split=data_split)
            if self.args.distributed:
                data_loader = data.DataLoader(data_set, batch_size=1, num_workers=8, drop_last=True, pin_memory=True,
                                              sampler=DistributedSampler(data_set))
            else:
                data_loader = data.DataLoader(data_set, batch_size=1, num_workers=8, drop_last=True, pin_memory=True,
                                              shuffle=False)
        return data_loader


    def saveModel_epoch(self):
        state = {'epoch': self.epoch,
                 'state_dict': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 }

        torch.save(state, os.path.join(self.output_path, f"{self.epoch}.pkl"))


    def train_one_epoch(self):
        self.model.train()
        for images, labels, segment in self.train_loader:
            self.iteration += 1
            images = Variable(images.cuda(self.args.device))
            labels = Variable(labels.cuda(self.args.device))
            segment = Variable(segment.cuda(self.args.device))

            outputs, pre_segment = self.model(images)
            loss = 0
            for loss_name in self.loss_dict.keys():
                if 'segment' in loss_name:
                    cur_loss = self.loss_dict[loss_name]['weight']*self.loss_dict[loss_name]['function'](pre_segment, segment)
                else:
                    cur_loss = self.loss_dict[loss_name]['weight']*self.loss_dict[loss_name]['function'](outputs, labels)
                loss += cur_loss
                self.loss_log.update(cur_loss.item(), loss_name, 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.iteration % self.args.print_freq == 0 or self.iteration == len(self.train_loader):
                if self.args.rank == 0:
                    print_str = '[{0}][{1}/{2}]\t\t'.format(self.epoch, self.iteration, len(self.train_loader))
                    for loss_name in self.loss_dict.keys():
                        print_str += '[{0}:{1:.4f}]\t'.format(loss_name, self.loss_log.Loss[loss_name]/self.loss_log.cnt)
                    print_str += '{0:.4f}'.format((self.loss_log.Loss['total'])/self.loss_log.cnt)
                    print(print_str)

                # print结束就重制，保证每次print结果等量级
                self.loss_log.reset()

        # 避免保存多次
        if self.args.rank == 0:
            self.saveModel_epoch()
        self.model.eval()
        if self.validate:
            self.validate_one_epoch()
        print("Current learning rate: ", self.optimizer.param_groups[0]['lr'])
        self.scheduler.step()

        print('\n')

    def validate_one_epoch(self):
        with torch.no_grad():
            for i_val, (images, labels, segment, img_name) in enumerate(self.val_loader):
                images = Variable(images.cuda(self.args.device))
                labels = Variable(labels.cuda(self.args.device))
                segment = Variable(segment.cuda(self.args.device))

                outputs, pred_segment = self.model(images)

                loss = 0
                for loss_name in self.loss_dict.keys():
                    if 'segment' in loss_name:
                        cur_loss = self.loss_dict[loss_name]['weight'] * self.loss_dict[loss_name]['function'](
                            pred_segment, segment)
                    else:
                        cur_loss = self.loss_dict[loss_name]['weight'] * self.loss_dict[loss_name]['function'](
                            outputs, labels)
                    loss += cur_loss
                    self.val_loss.update(cur_loss.item(), loss_name, 1)

            self.validate_image(images[0].cpu().numpy().transpose(1, 2, 0), outputs[0].cpu().numpy().transpose(1, 2, 0), labels[0].cpu().numpy().transpose(1,2,0), img_name[0].split('.')[0]+'.jpg')
        if self.args.rank == 0:
            print_str = '[{0}][{1}/{2}]\t\t'.format(self.epoch, self.iteration, len(self.train_loader))
            for loss_name in self.loss_dict.keys():
                print_str += '[{0}:{1:.4f}]\t'.format(loss_name, self.val_loss.Loss[loss_name] / self.val_loss.cnt)
            print_str += '{0:.4f}'.format((self.val_loss.Loss['total']) / self.val_loss.cnt)
            print(print_str)
        self.val_loss.reset()
        
    def train(self):
        if self.args.rank == 0:
            print("Begin training from epoch {} to epoch {}".format(self.epoch, self.args.n_epoch))
        while self.epoch < self.args.n_epoch:
            self.train_one_epoch()
            self.iteration = 0
            self.epoch += 1

    def validate_image(self, img, fiducial_points, labels, img_name):
        save_dir = os.path.join(self.output_path, 'val_images', str(self.epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        img = np.array(img, dtype=np.uint8)
        draw_img = img.copy()
        for i in range(len(fiducial_points)):
            for j in range(len(fiducial_points[i])):
                cv2.circle(draw_img, (int(fiducial_points[i][j][0]), int(fiducial_points[i][j][1])), 2, (0, 255, 0), -1)
        draw_img_2 = img.copy()
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                cv2.circle(draw_img_2, (int(labels[i][j][0]), int(labels[i][j][1])), 2, (0, 255, 0), -1)

        merged_img = np.zeros((img.shape[0], img.shape[1]*2, img.shape[2]), dtype=np.uint8)

        merged_img[:, :img.shape[1], :] = draw_img
        merged_img[:, img.shape[1]:, :] = draw_img_2
        cv2.imwrite(os.path.join(save_dir, img_name), merged_img)
