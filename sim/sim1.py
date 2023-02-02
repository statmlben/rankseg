# Author: Ben Dai <bendai@cuhk.edu.hk>
# Example 1 in Simulation section
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
from scipy.stats import truncnorm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# simulation
# sample_size, width, height, prob_type = 1000, 28, 28, 'exp'
def sim(base_=1.01, sample_size=1000, width=28, height=28, prob_type='exp'):
    num_class = 1
    dim = width * height

    ## produce prob matrix 
    if prob_type == 'exp':
        index_mat = torch.arange(width, device='cuda').repeat(height, 1)
        index_mat = index_mat + index_mat.T
        prob = base_ ** (-index_mat)
    elif prob_type == 'linear':
        index_mat = torch.arange(width, device='cuda').repeat(height, 1)
        index_mat = index_mat + index_mat.T
        prob = 1. - .5/width*base_*(index_mat)
        # prob = torch.where(prob < 0, 0, prob)
        prob[prob<0] = 0.
    elif prob_type == 'step':
        prob = truncnorm.rvs(0, 1, loc=base_, scale=0.1, size=(width, height))
        prob = torch.from_numpy(prob)
        prob = prob.cuda()
        prob = prob.view(width, height)
        # prob = 0.5*base_*torch.rand((width, height), device='cuda')
        prob[:int(.1*width), :int(.1*height)] = 0.5*torch.rand((int(.1*width), int(.1*height)), device='cuda') + 0.5
        prob[prob<0.] = 0.
        prob[prob>1.] = 1.
    prob = prob.view(num_class, width, height)

    ## generate sample
    target = torch.zeros((sample_size, num_class, width, height), device='cuda')
    for i in range(sample_size):
        target[i] = torch.bernoulli(prob)

    predict, _, _ = rank_dice(prob.repeat(1, 1, 1, 1), device='cuda', app=2, smooth=0., truncate_mean=False, verbose=0)
    predict_T = (prob > .5)

    score_rank = dice_coeff(predict.repeat(sample_size, 1, 1, 1), target)
    score_T = dice_coeff(predict_T.repeat(sample_size, 1, 1, 1), target)

    return [prob, score_rank, score_T, predict, predict_T]

sample_size = 2
for prob_type in ['step']:
    for base_ in [.1, .3, .5]:
        for width in [256]:
        # for width in [28, 64, 128, 256]:
            height = width
            prob, score_rank, score_T, pred_rd, pred_T = sim(base_=base_, sample_size=sample_size, width=width, height=height, prob_type=prob_type)
            print('#'*20)
            print('prob_type: %s; width: %d; height: %d' %(prob_type, width, height))
            print('dice score: rankdice: %.3f(%.3f); T: %.3f(%.3f)' %(score_rank.mean(), score_rank.std()/np.sqrt(sample_size), score_T.mean(), score_T.std()/np.sqrt(sample_size)))

        figure(figsize=(20, 15), dpi=100)
        ax = sns.heatmap(prob[0].cpu(), vmin=0, vmax=1, cmap="YlGnBu", cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])

        ## add box
        rectangles = {'28x28' : mpatch.Rectangle((0,0), 28, 28, linewidth=3, edgecolor='lightsteelblue', facecolor='none'),
                    '64x64' : mpatch.Rectangle((0,0), 64, 64, linewidth=3, edgecolor='cornflowerblue', facecolor='none'),
                    '128x128' : mpatch.Rectangle((0,0), 128, 128, linewidth=3, edgecolor='royalblue', facecolor='none'),
                    '256x256' : mpatch.Rectangle((0,0), 255.5, 255.5, linewidth=3, edgecolor='darkblue', facecolor='none'),
                    }

        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx + rectangles[r].get_width() - 12
            cy = ry + rectangles[r].get_height() - 4

            ax.annotate(r, (cx, cy), color='grey', weight='bold', 
                        fontsize=20, ha='center', va='center')
        plt.title('decay pattern: %s(%.3f)' %(prob_type, base_), fontsize=50)
        plt.tight_layout()
        plt.show()