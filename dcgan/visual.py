#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from utils  import Generator
import matplotlib.pyplot as plt
from IPython.display import HTML
import torchvision.utils as vutils
import matplotlib.animation as animation

from IPython import embed


if __name__ == "__main__":
    
    model_dir = '../checkpoints'
    mids = list(range(1, 11))
    fixed_noise = torch.randn(64, 100, 1, 1).cuda(0)
    generator = Generator(100, 64).cuda(0)
    generator = torch.nn.DataParallel(generator, device_ids=[0, 1])
    imgs_list = []
    
    for mid in mids:
        
        checkpoints = torch.load(os.path.join(model_dir, 'epoch_%d.pth.tar' % mid))
        epoch = checkpoints['epoch']
        generator.load_state_dict(checkpoints['generator'])
        print('epoch : %d, mid : %d' % (epoch, mid))    
        generator.eval()
        fake = generator(fixed_noise).detach().cpu()
        imgs_list.append(fake)
    
    
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    embed()
    ims = [[plt.imshow(np.transpose(i[0],(1, 2, 0)), animated=True)] for i in imgs_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1][0],(1,2,0)))
    plt.show()
    