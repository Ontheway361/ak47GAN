#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdb
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from IPython import embed

class LinearWeightNorm(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):

        super(LinearWeightNorm, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1


    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)


class Discriminator(nn.Module):

    def __init__(self, img_dim = 784, out_dim = 10):
        super(Discriminator, self).__init__()
        self.img_dim = img_dim
        self.layers = nn.ModuleList([
            LinearWeightNorm(img_dim, 1000),
            LinearWeightNorm(1000, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)]
        )
        self.final = LinearWeightNorm(250, out_dim, weight_scale=1)

    def forward(self, x, feature = False, cuda = False):
        x = x.view(-1, self.img_dim)
        noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor([0])
        if cuda:
            noise = noise.cuda()
        x = x + Variable(noise, requires_grad=False)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.relu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0])
            if cuda:
                noise = noise.cuda()
            x = (x_f + Variable(noise, requires_grad=False))
        if feature:
            return x_f, self.final(x)
        return self.final(x)


class Generator(nn.Module):

    def __init__(self, z_dim = 100, img_dim = 784):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 500, bias = False)
        self.bn1 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.bias1 = Parameter(torch.zeros(500))

        self.fc2 = nn.Linear(500, 500, bias = False)
        self.bn2 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.bias2 = Parameter(torch.zeros(500))
        self.fc3 = LinearWeightNorm(500, img_dim, weight_scale = 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, batch_size, cuda = False):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad = False, volatile = not self.training)
        if cuda:
            x = x.cuda()
        x = F.softplus(self.bn1(self.fc1(x)) + self.bias1)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bias2)
        x = F.softplus(self.fc3(x))
        return x
