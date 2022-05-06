import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer
import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
import numpy as np
from poibin import PoiBin
from scipy.stats import binom, norm
import scipy
from rankdice import rank_dice
from tester import Tester
import matplotlib.pyplot as plt
import seaborn as sns
## test for the new things

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device


    config['name'] = 'ENet'
    config['arch']['type'] = 'ENet'
    config['arch']['args']['backbone'] = 'resnet18'
    config['test_loader']['args']['batch_size'] = 8
    config['predict']['test'] = 'rankdice'
    resume = './saved/ENet/CE_DiceLoss/02-08_11-57/best_model.pth'

    test_logger = Logger()

    # DATA LOADERS
    test_loader = get_instance(dataloaders, 'test_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, test_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TESTING
    tester = Tester(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        test_loader=test_loader,
        test_logger=test_logger)

    device = tester.device

    # tester.test()
    with torch.no_grad():
        tester._reset_metrics()
        tbar = tqdm(tester.test_loader, ncols=130)
        data, target = next(iter(tester.test_loader))
        num_class = tester.num_classes
        output = tester.model(data)
        _, predict = torch.max(output.data, 1)
        # predict_rd, tau_rd, cutpoint_rd = rank_dice(output, app=2, device=tester.device)
        predict_rd, tau_rd, cutpoint_rd = rank_dice(output, app=1, alpha=.1, device=tester.device)
        out_prob = output.softmax(dim=1)
        predict_T = torch.where(out_prob > .5, True, False)

    ## cutpoint of rankdice
    ax = sns.heatmap(cutpoint_rd.data.cpu(), annot=True, fmt=".2f")
    plt.show()

    # vitualization
    val_img = []
    target_np, predict_np, predict_rd_np, predict_T_np = target.data.cpu(), predict.data.cpu(), predict_rd.data.cpu(), predict_T.data.cpu()
    val_visual = [[data[i].data.cpu(), target_np[i], predict_np[i], predict_rd_np[i], predict_T_np[i]] for i in range(len(target_np))]
    palette = tester.test_loader.dataset.palette
    for d, t, o, o_rd, o_T in val_visual:
        # d = tester.restore_transform(d)
        # d = d.convert('RGB')
        # d = tester.viz_transform(d)
        for k in range(tester.num_classes):
            t_tmp = torch.where(t==k, True, False)
            o_tmp = torch.where(o==k, True, False)
            o_rd_tmp = o_rd[k]
            o_T_tmp = o_T[k]
            
            if t_tmp.sum() >= 1:
                fig, axes = plt.subplots(1,5, figsize=(20,4))
                axes[0].imshow(d.permute(1, 2, 0))
                axes[1].imshow(t_tmp)
                axes[2].imshow(o_tmp)
                axes[3].imshow(o_rd_tmp)
                axes[4].imshow(o_T_tmp)
                axes[1].set_title("class-%s: true segmentation" %k)
                axes[2].set_title("class-%s: predicted (max) segmentation" %k)
                axes[3].set_title("class-%s: predicted (rankdice) segmentation" %k)
                axes[4].set_title("class-%s: predicted (T) segmentation" %k) 
                plt.show()
