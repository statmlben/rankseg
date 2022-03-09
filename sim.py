import os
import torch
from rankseg import rank_dice
from metrics import dice_coeff
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# simulation
# sample_size, width, height, prob_type = 1000, 28, 28, 'exp'
def sim(sample_size=1000, width=28, height=28, prob_type='exp'):
    num_class = 1
    dim = width * height

    ## produce prob matrix 
    if prob_type == 'exp':
        base_power = 1.7
        index_mat = torch.arange(width, device='cuda').repeat(height, 1)
        index_mat = index_mat + index_mat.T
        prob = base_power ** (-index_mat)
    elif prob_type == 'unif':
        prob = .4*torch.ones((width, height), device='cuda')
    prob = prob.view(num_class, width, height)

    ## generate sample
    target = torch.zeros((sample_size, num_class, width, height), device='cuda')
    for i in range(sample_size):
        target[i] = torch.bernoulli(prob)

    predict, _, _ = rank_dice(prob.repeat(1, 1, 1, 1), device='cuda', app=2, smooth=1., alpha=0.2, truncate_mean=False, verbose=0)
    predict_T = (prob > .5)

    score_rank = dice_coeff(predict.repeat(sample_size, 1, 1, 1), target)
    score_T = dice_coeff(predict_T.repeat(sample_size, 1, 1, 1), target)

    return [score_rank, score_T, predict, predict_T]

sample_size = 500
for prob_type in ['exp']:
    for width in [28, 64, 128, 256]:
    # for width in [28]:
        height = width
        score_rank, score_T, pred_rd, pred_T = sim(sample_size=sample_size, width=width, height=height, prob_type=prob_type)
        print('#'*20)
        print('prob_type: %s; width: %d; height: %d' %(prob_type, width, height))
        print('dice score: rankdice: %.3f; T: %.3f' %(score_rank.mean(), score_T.mean()))