import torch
import torch.nn.functional as F


class Losses(object):
    def __init__(self, k):
        self.k = k
        self.kernel_cross = torch.zeros(2, 1, self.k*2+1, self.k*2+1)
        self.kernel_cross[:, 0, self.k, :] = 1
        self.kernel_cross[:, 0, :, self.k] = 1


    def loss_regress(self, input, target):
        i_t = target - input

        '''one'''
        loss_l1 = F.smooth_l1_loss(input, target, reduction='mean')

        '''two'''
        self.kernel_cross = self.kernel_cross.to(target.device)
        local = torch.pow(F.conv2d(F.pad(i_t, (self.k, self.k, self.k, self.k), mode='replicate'), self.kernel_cross, padding=0, groups=2) - i_t*(self.k*4+1), 2)
        mask = F.conv2d(F.pad(torch.ones_like(i_t), (self.k, self.k, self.k, self.k), mode='constant', value=0), self.kernel_cross, padding=0, groups=2)
        mask = 1/mask*(self.k*4+1)
        loss_local = torch.mean(local*mask)
        return loss_l1 + 0.1*loss_local

    def loss_segment(self, input, target):
        return F.l1_loss(input, target, reduction='mean')

