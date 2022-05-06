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


    config['name'] = 'ENet'
    config['arch']['type'] = 'ENet'
    config['arch']['args']['backbone'] = 'resnet18'
    config['test_loader']['args']['batch_size'] = 8
    config['predict']['test'] = 'rankdice'
    config['predict']['test'] = 'rankdice'
    config['loss'] = 'BCEWithLogitsLoss2d'
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
        predict, tau_rd, cutpoint_rd = rank_dice(output, app=2, device=tester.device)
        eval_metrics(predict, target, num_class, smooth=1.)
        # print(eval_metrics(predict, target, num_class, smooth=1., panoptic=False))


###### TESTING ######

# ###### TESTING ######
# TEST, Pred (T) | Loss: 0.312, PixelAcc: 0.69, Mean IoU: 0.88, Mean Dice 0.88 |: 100%|███████████| 182/182 [00:29<00:00,  6.22it/s]

# ## TESTING Restuls for Model: ENet + Loss: DiceLoss + predict: T ## 
#     test_loss      : 0.31184
#     Pixel_Accuracy : 0.689
#     Mean_IoU       : 0.875
#     Mean_Dice      : 0.881
#     Class_IoU      : {0: 0.793, 1: 0.839, 2: 0.947, 3: 0.799, 4: 0.95, 5: 0.931, 6: 0.927, 7: 0.811, 8: 0.903, 9: 0.917, 10: 0.948, 11: 0.91, 12: 0.845, 13: 0.902, 14: 0.852, 15: 0.639, 16: 0.944, 17: 0.889, 18: 0.939, 19: 0.844, 20: 0.84}
#     Class_Dice     : {0: 0.87, 1: 0.845, 2: 0.947, 3: 0.799, 4: 0.95, 5: 0.931, 6: 0.927, 7: 0.814, 8: 0.904, 9: 0.917, 10: 0.948, 11: 0.911, 12: 0.847, 13: 0.902, 14: 0.856, 15: 0.668, 16: 0.944, 17: 0.89, 18: 0.939, 19: 0.848, 20: 0.844}

# TEST, Pred (rankdice) | Loss: 0.312, PixelAcc: 0.75, Mean IoU: 0.88, Mean Dice 0.89 |: 100%|████| 182/182 [01:27<00:00,  2.09it/s]

# ## TESTING Restuls for Model: ENet + Loss: DiceLoss + predict: rankdice ## 
#     test_loss      : 0.31184
#     Pixel_Accuracy : 0.746
#     Mean_IoU       : 0.877
#     Mean_Dice      : 0.885
#     Class_IoU      : {0: 0.795, 1: 0.836, 2: 0.947, 3: 0.796, 4: 0.951, 5: 0.931, 6: 0.926, 7: 0.809, 8: 0.918, 9: 0.917, 10: 0.949, 11: 0.912, 12: 0.859, 13: 0.911, 14: 0.856, 15: 0.643, 16: 0.944, 17: 0.892, 18: 0.939, 19: 0.841, 20: 0.834}
#     Class_Dice     : {0: 0.872, 1: 0.843, 2: 0.947, 3: 0.801, 4: 0.951, 5: 0.931, 6: 0.93, 7: 0.814, 8: 0.925, 9: 0.917, 10: 0.949, 11: 0.915, 12: 0.867, 13: 0.916, 14: 0.863, 15: 0.676, 16: 0.944, 17: 0.896, 18: 0.939, 19: 0.847, 20: 0.839}