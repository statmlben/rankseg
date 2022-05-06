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
### test for the new things test more test

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
    config['test_loader']['args']['batch_size'] = 2
    config['predict']['test'] = 'T'
    config['predict']['test'] = 'T'
    config['loss'] = 'BCEWithLogitsLoss2d'
    # config['loss'] = 'CrossEntropyLoss2d'

    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    test_loader = get_instance(dataloaders, 'test_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    # loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    loss = getattr(losses, config['loss'])()

    # resume = './saved/ENet/CE_DiceLoss/02-08_11-57/checkpoint-epoch500.pth'
    resume = None
    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    # trainer.train()

    tbar = tqdm(trainer.train_loader, ncols=130)
    data, target = next(iter(trainer.train_loader))
    num_class = trainer.num_classes
    
    output = trainer.model(data)
    target_oh = torch.zeros_like(output)
    for k in range(num_class):
        target_oh[:,k] = torch.where(target == k, 1, 0)

    loss = trainer.loss(output, target_oh)
    loss.backward()
    trainer.optimizer.step()


    