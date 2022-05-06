import os
import logging
import json
import math
import torch
import datetime
from torch.utils import tensorboard
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTester:
    def __init__(self, model, loss, resume, config, test_loader, test_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        self.test_loader = test_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_logger = test_logger
        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])


        cfg_tester = self.config['tester']
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        # TENSOBOARD
        writer_dir = os.path.join(cfg_tester['out_dir'], self.config['name'])
        self.writer = tensorboard.SummaryWriter(writer_dir)

        self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    
    def test(self):
        raise NotImplementedError
            
    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    
