import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from base import BaseTester, DataPrefetcher
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict
import time
from torchvision.utils import make_grid
from utils import transforms as local_transforms
from utils.metrics import eval_metrics, AverageMeter
from rankdice import rank_dice


class Tester(BaseTester):
    def __init__(self, model, loss, resume, config, test_loader, test_logger=None, prefetch=True):
        super(Tester, self).__init__(model, loss, resume, config, test_loader, test_logger)
        
        self.wrt_mode = 'test_'
        self.num_classes = self.test_loader.dataset.num_classes
        self.CoI = self.config['predict']['CoI']

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.test_loader.MEAN, self.test_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.test_loader = DataPrefetcher(test_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def test(self):
        if self.test_loader is None:
            self.logger.warning('Not data loader was passed for the test step, No testing is performed !')
            return {}
        self.logger.info('\n###### TESTING ######')

        self.model.eval()
        self.wrt_mode = 'test'

        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=130)
        with torch.no_grad():
            test_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                ## ignore the samples without any anotation
                if torch.all(target == self.config['ignore_index']):
                    print(batch_idx)
                    continue
                # data, target = next(iter(tester.test_loader))
                #data, target = data.to(self.device), target.to(self.device)
                assert self.config['predict']['test'] in {'max', 'T', 'rankdice'}

                output = self.model(data)

                if self.config['loss'][:3] == 'BCE':
                    target_oh = torch.zeros_like(output)
                    for k in range(self.num_classes):
                        target_oh[:,k] = torch.where(target == k, 1, 0)
                    target = target_oh

                ## LOSS
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                if self.config['predict']['test'] == 'max':
                    if self.config['loss'][:3] == 'BCE':
                        out_prob = output.sigmoid()
                        _, predict = torch.max(out_prob.data, 1)
                    else:
                        _, predict = torch.max(output.data, 1)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                elif self.config['predict']['test'] == 'T':
                    if self.config['loss'][:3] == 'BCE':
                        out_prob = output.sigmoid()
                    else:
                        out_prob = output.softmax(dim=1)
                    predict = torch.where(out_prob >= .5, True, False)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                else:
                    if self.config['loss'][:3] == 'BCE':
                        out_prob = output.sigmoid()
                    else:
                        out_prob = output.softmax(dim=1)
                    predict, _, _ = rank_dice(out_prob, app=2, device=self.device)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                
                self._update_seg_metrics(*seg_metrics)
                pixAcc, mIoU, mDice, cIoU, cDice = self._get_seg_metrics(batch_idx+1).values()

                # LIST OF IMAGE TO VIZ (15 images)
                if len(test_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    test_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                tbar.set_description('TEST, Pred ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}, Mean Dice {:.2f} |'.format(self.config['predict']['test'],
                                                self.total_loss.average,
                                                pixAcc, mIoU, mDice))

            # WRTING & VISUALIZING THE MASKS
            # test_img = []
            # palette = self.test_loader.dataset.palette
            # for d, t, o in test_visual:
            #     d = self.restore_transform(d)
            #     t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            #     d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            #     [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            #     test_img.extend([d, t, o])
            # test_img = torch.stack(test_img, 0)
            # test_img = make_grid(test_img.cpu(), nrow=3, padding=5)
            # self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', test_img)

            # METRICS TO TENSORBOARD
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average)
            seg_metrics = self._get_seg_metrics(batch_idx+1)
            # seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-2]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v)

            log = {
                'test_loss': self.total_loss.average,
                **seg_metrics
            }
            
            self.logger.info(f'\n    ## TESTING Restuls for Model: %s + Loss: %s + predict: %s ## ' %(self.config['name'], self.config['loss'], self.config['predict']['test']))
            for k, v in log.items():
                self.logger.info(f'         {str(k):15s}: {v}')

        return log
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        # self.total_inter, self.total_union, self.total_union_sum = 0, 0, 0
        # self.total_correct, self.total_label = 0, 0
        self.sPixAcc, self.sIoU, self.sDice, = [], [], []
        self.smIoU, self.smDice = [], []

    def _update_seg_metrics(self, pixAcc, mIoU, mDice, cIoU, cDice):
        # self.total_correct += correct
        # self.total_label += labeled
        # self.total_inter += inter
        # self.total_union += union
        # self.total_union_sum += union_sum
        self.sPixAcc.extend(pixAcc)
        self.sIoU.extend(cIoU)
        self.sDice.extend(cDice)
        self.smDice.extend(mDice)
        self.smIoU.extend(mIoU)

    def _get_seg_metrics(self, num_batch):
        # pixAcc = 1.0 * self.total_correct / (np.spacing(1e-5) + self.total_label)
        # IoU = 1.0 * self.total_inter / (np.spacing(1e-5) + self.total_union)
        # Dice = 2.0 * self.total_inter / (np.spacing(1e-5) + self.total_union_sum)
        # mIoU = IoU.mean()
        # mDice = Dice.mean()
        # pixAcc = self.sPixAcc / num_batch
        # IoU = self.sIoU / num_batch
        # Dice = self.sDice / num_batch
        # mIoU = self.smIoU / num_batch
        # mDice = self.smDice / num_batch
        pixAcc = np.nanmean(self.sPixAcc)
        IoU = np.nanmean(self.sIoU, axis=0)
        Dice = np.nanmean(self.sDice, axis=0)
        mIoU = np.nanmean(self.smIoU)
        mDice = np.nanmean(self.smDice)
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Mean_Dice": np.round(mDice, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
            "Class_Dice": dict(zip(range(self.num_classes), np.round(Dice, 3)))
        }