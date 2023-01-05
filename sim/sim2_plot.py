# Author: Ben Dai <bendai@cuhk.edu.hk>
# Plot of Example 2 in Simulation section
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

# simulation: plot Example 2
def sim(beta, rho, width=64, height=64):
    num_class = 1
    dim = width * height

    prob = truncnorm.rvs(0, 1, loc=beta, scale=0.1, size=(width, height))
    prob = torch.from_numpy(prob)
    prob = prob.cuda()
    prob = prob.view(width, height)
    prob[:int(rho*width), :int(rho*height)] = 0.5*torch.rand((int(rho*width), int(rho*height)), device='cuda') + 0.5
    prob[prob<0.] = 0.
    prob[prob>1.] = 1.
    prob = prob.view(num_class, width, height)

    predict, _, cutpoint_rd = rank_dice(prob.repeat(1, 1, 1, 1), device='cuda', app=2, smooth=0., truncate_mean=False, pruning=False, verbose=0)

    return cutpoint_rd

df = {'beta': [], 'rho': [], 'opt_THOLD': []}
for beta in [.1, .2, .3, .4, .5, .6, .7]:
    for rho in [.1, .2, .3, .4, .5, .6, .7, .8]:
        cutpoint_tmp = sim(beta=beta, rho=rho, width=64, height=64)
        cutpoint_tmp = cutpoint_tmp.cpu().numpy()[0][0]
        print('beta: %.3f; rho: %.3f; cut: %.3f' %(beta, rho, cutpoint_tmp))
        df['beta'].append(beta)
        df['rho'].append(rho)
        df['opt_THOLD'].append(cutpoint_tmp)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(15, 15), dpi=100)
df = pd.DataFrame(df)
df_heat = df.pivot("beta", "rho", "opt_THOLD")

sns.heatmap(df_heat, annot=True, cmap="YlGnBu", fmt=".1f", cbar=False)
plt.xlabel(r'$\beta$ (noise level)', fontsize=20)
plt.ylabel(r'$\rho$ (simulated segmentation proportion)', fontsize=20)
plt.tight_layout()
plt.show()