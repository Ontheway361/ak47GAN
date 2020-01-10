#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

from utils import *

from IPython import embed


class DCGAN(object):

    def __init__(self, args):
        self.args = args
        self.model  = dict()
        self.data   = dict()
        self.result = list()
        self.device = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python    : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.device))
        print('-' * 52)


    def _model_loader(self):

        self.model['generator'] = Generator(self.args.in_dim, self.args.gchannels)
        self.model['discriminator'] = Discriminator(self.args.dchannels)
        self.model['criterion'] = nn.BCELoss()
        self.model['opti_gene'] = optim.Adam(self.model['generator'].parameters(), \
                                                 lr=self.args.base_lr, betas=(self.args.beta, 0.999))
        self.model['opti_disc'] = optim.Adam(self.model['discriminator'].parameters(), \
                                                 lr=self.args.base_lr, betas=(self.args.beta, 0.999))
        # self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
        #                               self.model['optimizer'], milestones=[12, 20, 30, 45], gamma=self.args.gamma)
        if self.device:
            self.model['generator'] = self.model['generator'].cuda()
            self.model['discriminator'] = self.model['discriminator'].cuda()
            if len(self.args.gpu_ids) > 1:
                self.model['generator'] = torch.nn.DataParallel(self.model['generator'], device_ids=self.args.gpu_ids)
                self.model['discriminator'] = torch.nn.DataParallel(self.model['discriminator'], device_ids=self.args.gpu_ids)
                torch.backends.cudnn.benchmark = True
                print('Parallel mode was going ...')
            else:
                print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start = checkpoint['epoch']
            self.model['generator'].load_state_dict(checkpoint['generator'])
            self.model['discriminator'].load_state_dict(checkpoint['discriminator'])
            print('Resuming the train process at %3d epoches ...' % self.args.start)
        print('Model loading was finished ...')


    def _data_loader(self):

        self.data['train_loader'] = DataLoader(
                                        CelebA(args=self.args),
                                        batch_size = self.args.batch_size, \
                                        shuffle    = True,\
                                        num_workers= self.args.workers)
        self.data['fixed_noise'] = torch.randn(64, self.args.in_dim ,1, 1)
        if self.device:
            self.data['fixed_noise'] = self.data['fixed_noise'].cuda()
        print('Data loading was finished ...')


    def _model_train(self, epoch = 0):

        total_dloss, total_gloss = 0, 0
        for idx, imgs in enumerate(self.data['train_loader']):


            self.model['discriminator'].train()
            self.model['generator'].eval()
            imgs.requires_grad = False
            if self.device:
                imgs = img.cuda()
            b_size = imgs.size(0)
            self.model['discriminator'].zero_grad()
            gty = torch.full((b_size,), 1)
            if self.device:
                gty = gty.cuda()
            predy = self.model['discriminator'](imgs).view(-1)
            dloss_real = self.model['criterion'](predy, gty)
            dloss_real.backward()

            noise = torch.randn(b_size, self.args.in_dim, 1, 1)
            if self.device:
                noise = noise.cuda()
            fake = self.model['generator'](noise)
            gty.fill_(0)
            predy = self.model['discriminator'](fake.detach()).view(-1)
            dloss_fake = self.model['criterion'](predy, gty)
            dloss_fake.backward()
            self.model['opti_disc'].step()

            self.model['generator'].train()
            self.model['discriminator'].eval()
            self.model['generator'].zero_grad()
            gty.fill_(1)
            predy = self.model['discriminator'](fake).view(-1)
            gloss = self.model['criterion'](predy, gty)
            gloss.backward()
            self.model['opti_gene'].step()

            d_loss_real = dloss_real.mean().item()
            d_loss_fake = dloss_fake.mean().item()
            d_loss = d_loss_real + d_loss_fake
            g_loss = gloss.mean().item()
            total_dloss += d_loss
            total_gloss += g_loss
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : [%2d|%2d], iter : %4d, d_loss_real : %.4f, d_loss_fake : %.4f, \
                       d_loss : %.4f, g_loss : %.4f' % (epoch, self.epoches, idx+1, d_loss_real, \
                       d_loss_fake, d_loss, g_loss))

            if (idx + 1) % self.args.monitor_freq == 0:
                with torch.no_grad():
                    fake = self.model['generator'](self.data['fixed_noise']).detach().cpu()
                self.result.append(fake)

        return total_dloss, total_gloss


    def _main_loop(self):

        min_loss = 1e3
        for epoch in range(self.args.start, self.args.epoches + 1):

            start_time = time.time()
            dloss, gloss = self._model_train(epoch)
            train_loss = dloss + gloss
            # self.model['scheduler'].step()
            end_time = time.time()
            print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))

            if not os.path.exists(self.args.save_to):
                os.mkdir(self.args.save_to)

            if min_loss > train_loss:
                print('%snew SOTA was found%s' % ('*'*16, '*'*16))
                min_loss = train_loss
                filename = os.path.join(self.args.save_to, 'sota.pth.tar')
                torch.save({
                    'epoch'         : epoch,
                    'generator'     : self.model['generator'].state_dict(),
                    'discriminator' : self.model['discriminator'].state_dict(),
                    'loss'          : min_loss,
                }, filename)

            if epoch % self.args.save_freq == 0:
                filename = os.path.join(self.args.save_to, 'epoch_'+str(epoch)+'.pth.tar')
                torch.save({
                    'epoch'         : epoch,
                    'generator'     : self.model['generator'].state_dict(),
                    'discriminator' : self.model['discriminator'].state_dict(),
                    'loss'          : train_loss,
                }, filename)

            if self.args.is_debug:
                break


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    faceu = DCGAN(training_args())
    faceu.train_runner()
