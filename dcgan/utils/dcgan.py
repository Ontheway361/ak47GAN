#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

__all__ = ['Generator', 'Discriminator']

class Generator(nn.Module):

    def __init__(self, in_dim = 100, gchannels = 64):

        super(Generator, self).__init__()
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(in_dim, gchannels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gchannels * 8),
            nn.ReLU(True),
            # state size. (gf_channels*8) x 4 x 4
            nn.ConvTranspose2d(gchannels * 8, gchannels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gchannels * 4),
            nn.ReLU(True),
            # state size. (gf_channels*4) x 8 x 8
            nn.ConvTranspose2d(gchannels * 4, gchannels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gchannels * 2),
            nn.ReLU(True),
            # state size. (gf_channels*2 x 16 x 16
            nn.ConvTranspose2d(gchannels * 2, gchannels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gchannels),
            nn.ReLU(True),
            # state size. gf_channels x 32 x 32
            nn.ConvTranspose2d(gchannels, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.backbone(input)


class Discriminator(nn.Module):

    def __init__(self, dchannels = 64):

        super(Discriminator, self).__init__()
        self.backbone = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, dchannels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. dchannels x 32 x 32
            nn.Conv2d(dchannels, dchannels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dchannels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(dchannels * 2, dchannels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dchannels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dchannels * 4, dchannels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dchannels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(dchannels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.backbone(input)
