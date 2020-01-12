#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from IPython import embed

__all__ = ['CelebA']

class CelebA(data.Dataset):

    def __init__(self, args):

        self.args  = args
        self.transform = transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.Resize(args.img_size),
                             transforms.CenterCrop(args.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])
        self.fileslist = os.listdir(args.data_dir)
        if args.is_debug:
            self.fileslist = self.fileslist[:1024]


    def __getitem__(self, index):

        try:
            img_path = os.path.join(self.args.data_dir, self.fileslist[index])
            img = cv2.imread(img_path)
        except:
            idx = min(max(0, index - 1), len(self.fileslist))
            img_path = os.path.join(self.args.data_dir, self.fileslist[idx])
            img = cv2.imread(img_path)
        img = self.transform(img)
        return img


    def __len__(self):
        return len(self.fileslist)


root_dir = '/Users/relu/data/benchmark_images/ak47/align_celeba'

def config():

    parser = argparse.ArgumentParser('Test for CelebA')
    parser.add_argument('--data_dir',  type=str,  default=root_dir)
    parser.add_argument('--img_size',  type=int,  default=64)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    dataloader = data.DataLoader(
                     CelebA(args=config()),
                     batch_size = 128,
                     shuffle    = True,
                     num_workers= 2)

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    embed()
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0], \
                                padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('train.jpg', dpi=400)
    plt.close()
