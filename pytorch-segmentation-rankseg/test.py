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
from tester import Tester

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    # resume = './saved/VOC/DeepLab/CrossEntropyLoss2d/max/03-29_16-20/best_model.pth'
    # config = './saved/VOC/DeepLab/CrossEntropyLoss2d/max/03-29_16-20/config.json'; config = json.load(open(config))
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

    tester.test()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-p', '--predict', default=None, type=str,
                           help='how to make prediction after obtaining the probs')
    # parser.add_argument('-l', '--CoI', default=None, type=list,
    #                        help='classes of interests')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.predict:
        config['predict']['test'] = args.predict
        # config['predict']['CoI'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)

