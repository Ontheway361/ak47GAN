#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path as osp

root_dir = '/Users/relu/data/benchmark_images/ak47/align_celeba'

def training_args():

    parser = argparse.ArgumentParser('Config for DCGAN')

    # platform
    parser.add_argument('--use_gpu',  type=bool, default=False)
    parser.add_argument('--gpu_ids',  type=list, default=[0, 1])
    parser.add_argument('--workers',  type=int,  default=4)

    # model
    parser.add_argument('--in_dim',    type=int, default=100)
    parser.add_argument('--gchannels', type=int, default=64)
    parser.add_argument('--dchannels', type=int, default=64)

    # dataset
    parser.add_argument('--data_dir',   type=str, default=root_dir)
    parser.add_argument('--img_size',   type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)  # default = 128

    # optimize
    parser.add_argument('--epoches', type=int,   default=5)
    parser.add_argument('--start',   type=int,   default=1)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--beta',    type=float, default=0.5)
    parser.add_argument('--resume',  type=str,   default='')

    # verbose
    parser.add_argument('--save_freq',    type=int, default=10)
    parser.add_argument('--print_freq',   type=int, default=500)
    parser.add_argument('--monitor_freq', type=int, default=2000)
    parser.add_argument('--is_debug',     type=bool,default=False)
    parser.add_argument('--save_to',      type=str, default='../checkpoints')
    args = parser.parse_args()
    return args
