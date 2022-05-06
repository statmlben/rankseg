# https://pythonawesome.com/semantic-segmentation-models-datasets-and-losses-implemented-in-pytorch/
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
from rankdice import rank_dice

class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')
        
        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'
        self.CoI = self.config['predict']['CoI']

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            #data, target = data.to(self.device), target.to(self.device)
            self.lr_scheduler.step(epoch=epoch-1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)

            if self.config['loss'][:3] == 'BCE':
                if self.config['arch']['type'][:3] == 'PSP':
                    target_oh = torch.zeros_like(output[0])
                else:
                    target_oh = torch.zeros_like(output)
                for k in range(self.num_classes):
                    target_oh[:,k] = torch.where(target == k, 1, 0)
                target = target_oh

            if self.config['arch']['type'][:3] == 'PSP':
                # assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes 
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                # assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes 
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
                
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            # seg_metrics = eval_metrics(output, target, self.num_classes)
            # self._update_seg_metrics(*seg_metrics)
            # pixAcc, mIoU, mDice, cIoU, cDice = self._get_seg_metrics().values()

            # FOR UPDATED EVAL
            assert self.config['predict']['train'] in {'max', 'T', 'rankdice'}

            with torch.no_grad():
                output = self.model(data)
                if self.config['arch']['type'][:3] == 'PSP':
                    output = output[0]
                if self.config['predict']['train'] == 'max':
                    _, predict = torch.max(output.data, 1)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                elif self.config['predict']['train'] == 'T':
                    if self.config['loss'][:3] == 'BCE':
                        out_prob = output.sigmoid()
                    else:
                        out_prob = output.softmax(dim=1)
                    predict = torch.where(out_prob > .5, True, False)
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

            # PRINT INFO
            tbar.set_description('TRAIN ({}) Pred ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} mDice {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.config['predict']['train'], self.total_loss.average, 
                                                pixAcc, mIoU, mDice,
                                                self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics(batch_idx+1)
        for k, v in list(seg_metrics.items())[:-2]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                # output = self.model(data)
                # loss = self.loss(output, target)
                # if isinstance(self.loss, torch.nn.DataParallel):
                #     loss = loss.mean()
                # self.total_loss.update(loss.item())

                assert self.config['predict']['train'] in {'max', 'T', 'rankdice'}
                
                output = self.model(data)

                if self.config['loss'][:3] == 'BCE':
                    target_oh = torch.zeros_like(output)
                    for k in range(self.num_classes):
                        target_oh[:,k] = torch.where(target == k, 1, 0)
                    target = target_oh

                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                if self.config['predict']['val'] == 'max':
                    _, predict = torch.max(output.data, 1)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                elif self.config['predict']['val'] == 'T':
                    if self.config['loss'][:3] == 'BCE':
                        out_prob = output.sigmoid()
                    else:
                        out_prob = output.softmax(dim=1)
                    predict = torch.where(out_prob > .5, True, False)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                else:
                    if self.config['loss'][:3] == 'BCE':
                        out_prob = output.sigmoid()
                    else:
                        out_prob = output.softmax(dim=1)
                    predict, _, _ = rank_dice(out_prob, app=2, device=self.device)
                    seg_metrics = eval_metrics(predict, target, self.num_classes, self.CoI)
                
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, mDice, cIoU, cDice = self._get_seg_metrics(batch_idx+1).values()
                tbar.set_description('EVAL ({}), Pred ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f}, Mean Dice {:.2f} |'.format( epoch,
                                                self.config['predict']['val'], self.total_loss.average,
                                                pixAcc, mIoU, mDice))

            # WRTING & VISUALIZING THE MASKS
            # val_img = []
            # palette = self.train_loader.dataset.palette
            # for d, t, o in val_visual:
            #     d = self.restore_transform(d)
            #     t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            #     d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            #     [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            #     val_img.extend([d, t, o])
            # val_img = torch.stack(val_img, 0)
            # val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            # self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics(batch_idx+1)
            for k, v in list(seg_metrics.items())[:-2]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

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
