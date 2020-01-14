#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

root_dir = '/home/jovyan/lujie/gpu3/benchmark_images/ak47/align_celeba' # gpu-server

def training_args():
    parser = argparse.ArgumentParser(description='PyTorch for Improved GAN')

    # platform
    parser.add_argument('--use_gpu',  type=bool, default=False)   # TODO
    parser.add_argument('--gpu_ids',  type=list, default=[0, 1])
    parser.add_argument('--workers',  type=int,  default=4)

    # model
    parser.add_argument('--z_dim',    type=int, default=100)
    parser.add_argument('--img_dim',  type=int, default=784) # 28 * 28
    parser.add_argument('--out_dim',  type=int, default=10)

    # dataset
    parser.add_argument('--data_dir',   type=str, default=root_dir)
    parser.add_argument('--batch_size', type=int, default=100)  # num_images = 202599 | 128~1582
    parser.add_argument('--class_num',  type=int, default=10)

    # optimize
    parser.add_argument('--epochs',  type=int,   default=10)
    parser.add_argument('--start',   type=int,   default=1)
    parser.add_argument('--base_lr', type=float, default=3e-3)
    parser.add_argument('--beta',    type=float, default=0.5)
    parser.add_argument('--resume',  type=str,   default='')
    parser.add_argument('--wusup',   type=float, default=1.0)

    # verbose
    parser.add_argument('--save_freq',    type=int, default=1)     # TODO
    parser.add_argument('--log_freq',     type=int, default=100)
    parser.add_argument('--logdir',       type=str, default='./logfile')
    parser.add_argument('--is_debug',     type=bool,default=False)   # TODO
    parser.add_argument('--savedir',      type=str, default='./checkpoint')
    args = parser.parse_args()
    return args
