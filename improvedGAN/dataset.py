#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

root_dir = '/Users/relu/data/deep_learning/cs231n/benchmark' # mini-mac

def MnistSup(class_num):
    raw_dataset = datasets.MNIST(root_dir, train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))


def MnistUsup():
    return datasets.MNIST(root_dir, train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))


def MnistTest():
    return datasets.MNIST(root_dir, train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))


if __name__ == '__main__':
    print(dir(MnistTest()))
