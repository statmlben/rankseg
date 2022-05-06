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

    test_logger = Logger()

    # DATA LOADERS
    test_loader = get_instance(dataloaders, 'test_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, test_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    resume = 'saved/cityscapes/DeepLab/CrossEntropyLoss2d/T/04-07_15-47/best_model.pth'
    # resume = 'saved/DeepLab/BCEWithLogitsLoss2d/rankdice/03-11_13-55/best_model.pth'

    # TESTING
    tester = Tester(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        test_loader=test_loader,
        test_logger=test_logger)

    device = tester.device
    num_visual = 10
    # tester.test()
    with torch.no_grad():
        tester._reset_metrics()
        tbar = tqdm(tester.test_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            if batch_idx == num_visual:
                break
            # data, target = next(iter(tester.test_loader))
            target_tmp = torch.where(target==0, True, False)
            num_class = tester.num_classes
            output = tester.model(data)
            prob_out = output.sigmoid()
            predict_T = torch.where(prob_out>.5, True, False)
            predict, tau_rd, cutpoint_rd = rank_dice(output, app=2, device=tester.device, truncate_mean=True, verbose=0)
            pixAcc, mIoU, mDice, IoU_db, Dice_db = eval_metrics(predict, target, num_class, smooth=1., verbose=0)
            print('Ground Truth: %s' %target_tmp.sum((-1,-2)).cpu())
            print('predict T   : %s' %predict_T.sum((-1,-2)).cpu()[:,0])
            print('predict rd  : %s' %predict.sum((-1,-2)).cpu()[:,0])
            print('cutpoint_rd : %s' %cutpoint_rd.cpu()[:,0])


