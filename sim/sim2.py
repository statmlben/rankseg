# Author: Ben Dai <bendai@cuhk.edu.hk>
# Example 2 in Simulation section
# License: BSD 3 clause

import os
import torch
import sys
sys.path.append('../')

from rankseg import rank_dice
from metrics import dice_coeff
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatch
from torch.distributions import Beta
from scipy.stats import truncnorm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# simulation: Example 2
def sim(n_images=1000, sample_size=100, width=28, height=28):
    num_class = 1
    dim = width * height
    score_rank_lst, score_T_lst, cutpoint_lst = [], [], []

    for i in range(n_images):
        rho = np.random.rand()

        prob = truncnorm.rvs(0, 1, loc=0.1, scale=0.1, size=(width, height))
        prob = torch.from_numpy(prob)
        prob = prob.cuda()
        prob = prob.view(width, height)
        prob[:int(rho*width), :int(rho*height)] = 0.5*torch.rand((int(rho*width), int(rho*height)), device='cuda') + 0.5
        prob[prob<0.] = 0.
        prob[prob>1.] = 1.
        prob = prob.view(num_class, width, height)

        ## generate sample
        target = torch.zeros((sample_size, num_class, width, height), device='cuda')
        for i in range(sample_size):
            target[i] = torch.bernoulli(prob)

        predict, _, cutpoint_rd = rank_dice(prob.repeat(1, 1, 1, 1), device='cuda', app=2, smooth=0., truncate_mean=False, pruning=False, verbose=0)
        score_rank = dice_coeff(predict.repeat(sample_size, 1, 1, 1), target)

        score_T = []
        for prob_thresh in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            predict_T = (prob > prob_thresh)
            score_T_tmp = dice_coeff(predict_T.repeat(sample_size, 1, 1, 1), target)
            score_T.append(score_T_tmp.mean().cpu().numpy())
        
        score_rank_lst.append(score_rank.mean().cpu().numpy())
        score_T_lst.append(score_T)
        cutpoint_lst.append(cutpoint_rd[0][0].cpu().numpy())
    return np.array(score_rank_lst), np.array(score_T_lst), np.array(cutpoint_lst)

n_images = 2000
for width in [64]:
    height = width
    score_rank, score_T, cutpoint = sim(n_images=n_images, sample_size=100, width=width, height=height)
    print('#'*20)
    print('width: %d; height: %d' %(width, height))
    print('dice score: rankdice: %.3f(%.3f)\n T: %s(%s)' %(score_rank.mean(), score_rank.std()/np.sqrt(n_images), score_T.mean(axis=0), score_T.std(axis=0)/np.sqrt(n_images)))
