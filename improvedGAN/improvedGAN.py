#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
import tensorboardX
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset

from dataset import *
from config import training_args
from model import Generator, Discriminator

from IPython import embed


class ImprovedGAN(object):

    def __init__(self, args):

        self.args   = args
        self.data   = dict()
        self.model  = dict()
        self.writer = tensorboardX.SummaryWriter(log_dir=args.logdir)
        self._model_loader()
        self._data_loader()


    def _model_loader(self):

        self.model['generator'] = Generator(self.args.z_dim, self.args.img_dim)
        self.model['discriminator'] = Discriminator(self.args.img_dim, self.args.out_dim)
        self.model['optiD'] = optim.Adam(self.model['generator'].parameters(), \
                                             lr=self.args.base_lr, betas=(self.args.beta, 0.999))
        self.model['optiG'] = optim.Adam(self.model['discriminator'].parameters(), \
                                             lr=self.args.base_lr, betas=(self.args.beta, 0.999))

        if self.args.use_gpu:
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
            print('Resuming the train process at %2d epoches ...' % self.args.start)
        print('model loading was finished ...')


    def _data_loader(self):

        self.data['sup']  = MnistSup(self.args.class_num)
        self.data['usup'] = DataLoader(MnistUsup(), \
                                 batch_size = self.args.batch_size, shuffle = True, \
                                 drop_last  = True, num_workers= self.args.workers)
        self.data['test'] = DataLoader(MnistTest(), \
                                 batch_size = self.args.batch_size, \
                                 num_workers= self.args.workers)

        times = int(np.ceil(len(self.data['usup']) / self.data['sup'].__len__()))
        sup_x = self.data['sup'].tensors[0].repeat(times, 1, 1, 1)
        sup_y = self.data['sup'].tensors[1].repeat(times)
        self.data['sup'] = DataLoader(TensorDataset(sup_x, sup_y),\
                               batch_size = self.args.batch_size, shuffle = True, \
                               drop_last  = True, num_workers= self.args.workers)
        print('data loading was finished ...')


    def trainD(self, sup_imgs, sup_y, usup_imgs):

        sup_imgs  = Variable(sup_imgs, requires_grad=False)
        sup_y     = Variable(sup_y, requires_grad=False)
        usup_imgs = Variable(usup_imgs, requires_grad = False)

        if self.args.use_gpu:
            sup_imgs, usup_imgs, sup_y = sup_imgs.cuda(), usup_imgs.cuda(), sup_y.cuda()
        out_sup  = self.model['discriminator'](sup_imgs, self.args.use_gpu)
        out_usup = self.model['discriminator'](usup_imgs, self.args.use_gpu)
        finput   = self.model['generator'](usup_imgs.size()[0], self.args.use_gpu)
        out_fake = self.model['discriminator'](finput.view(usup_imgs.size()).detach(), self.args.use_gpu)

        logz_sup  = torch.log(torch.sum(torch.exp(out_sup),  dim=1))
        logz_usup = torch.log(torch.sum(torch.exp(out_usup), dim=1))
        logz_fake = torch.log(torch.sum(torch.exp(out_fake), dim=1))

        logit_sup = torch.gather(out_sup, 1, sup_y.unsqueeze(1))
        loss_sup  = - torch.mean(logit_sup) + torch.mean(logz_sup)  # softmax-loss
        loss_real = - torch.mean(logz_usup) + torch.mean(F.softplus(logz_usup))  # log Z/(1+Z)
        loss_fake = torch.mean(F.softplus(logz_fake)) # log 1/(1+Z)
        loss_usup = (loss_real + loss_fake) / 2
        loss = loss_sup + self.args.wusup * loss_usup
        acc_sup = torch.mean((out_sup.max(dim=1)[1] == sup_y).float())
        self.model['optiD'].zero_grad()
        loss.backward()
        self.model['optiD'].step()
        return loss_sup.item(), loss_usup.item(), acc_sup


    def trainG(self, x_usup):

        fake = self.model['generator'](x_usup.size()[0], self.args.use_gpu).view(x_usup.size())
        gfeat, _ = self.model['discriminator'](fake, True, self.args.use_gpu)
        dfeat, _ = self.model['discriminator'](Variable(x_usup), True, self.args.use_gpu)
        feat_diff = torch.mean(gfeat, dim=0) - torch.mean(dfeat, dim=0)
        loss = torch.mean(feat_diff**2)
        self.model['optiG'].zero_grad()
        self.model['optiD'].zero_grad()
        loss.backward()
        self.model['optiG'].step()
        return loss.item()


    def train(self):

        global_count = 0
        for epoch in range(self.args.epochs):
            self.model['generator'].train()
            self.model['discriminator'].train()

            cum_loss_sup, cum_loss_usup = 0, 0
            cum_loss_gen, cum_acc_sup = 0, 0
            batch_num = 0
            for idx, (usup_imgs, _) in enumerate(self.data['usup']):

                sup_imgs, sup_y = next(iter(self.data['sup']))
                loss_sup, loss_usup, acc_sup = self.trainD(sup_imgs, sup_y, usup_imgs)
                cum_loss_sup  += loss_sup
                cum_loss_usup += loss_usup
                cum_acc_sup   += acc_sup
                loss_g = self.trainG(usup_imgs)
                if epoch > 1 and lg > 1:
                    loss_g = self.trainG(usup_imgs)
                cum_loss_gen += loss_g
                if (idx + 1) % self.args.log_freq == 0:
                    print('Training: %d / %d' % (idx + 1, len(self.data['usup'])))
                    global_count += 1
                    with torch.no_grad():
                        real_feat = self.model['discriminator'](Variable(sup_imgs), True, self.args.use_gpu)[0]
                        fake_imgs = self.model['generator'](self.args.batch_size, self.args.use_gpu)
                        fake_feat = self.model['discriminator'](fake_imgs, True, self.args.use_gpu)[0]
                        self.writer.add_scalars('loss', {'loss_sup':loss_sup, 'loss_usup':loss_usup, 'loss_g':loss_g}, global_count)
                        self.writer.add_histogram('real_feature', real_feat, global_count)
                        self.writer.add_histogram('fake_feature', fake_feat, global_count)
                        self.writer.add_histogram('fc3_bias', self.model['generator'].fc3.bias, global_count)
                        self.writer.add_histogram('D_feature_weight', self.model['discriminator'].layers[-1].weight, global_count)
                    self.model['discriminator'].train()
                    self.model['generator'].train()
            cum_loss_sup  /= idx
            cum_loss_usup /= idx
            cum_loss_gen  /= idx
            cum_acc_sup   /= idx
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % \
                  (epoch, cum_loss_sup, cum_loss_usup, cum_loss_gen, cum_acc_sup))
            sys.stdout.flush()
            eval_acc = self.eval()
            print("Eval_acc : %,4f"  % eval_acc)
            if (epoch + 1) % self.args.save_freq == 0:
                if not os.path.exists(self.args.savedir):
                    os.mkdir(self.args.savedir)
                save_name = os.path.join(self.args.savedir, 'ImprovedGAN_epoch_%d.pth.tar' % epoch)
                torch.save({
                    'epoch'     : epoch,
                    'generator' : self.model['generator'].state_dict(),
                    'discriminator' : self.model['discriminator'].state_dict(),
                    'eval_acc'  : eval_acc,
                }, save_name)


    def eval(self):
        self.model['generator'].eval()
        self.model['discriminator'].eval()
        predy, gty = [], []
        with torch.no_grad():
            for idx, (imgs, y) in enumerate(self.data['test']):
                if self.args.use_gpu:
                    imgs = imgs.cuda()
                pred = self.model['discriminator'](Variable(imgs), self.args.use_gpu)
                pred.extend(torch.max(pred, dim=1)[1].cpu().numpy().tolist())  # BUG
                gty.extend(y.numpy().tolist())
            eval_acc = np.mean(np.array(predy) == np.array(gty))
        return eval_acc

    def draw(self, batch_size):
        self.G.eval()
        return self.model['generator'](batch_size, cuda=self.args.use_gpu)


if __name__ == '__main__':

    gan = ImprovedGAN(training_args())
    gan.train()
