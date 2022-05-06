import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

# def batch_pix_accuracy(predict, target, labeled):
#     pixel_labeled = labeled.sum()
#     pixel_correct = ((predict == target) * labeled).sum()
#     assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
#     return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

# def batch_intersection_union(predict, target, num_class, labeled):
#     predict = predict * labeled.long()
#     intersection = predict * (predict == target).long()

#     area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
#     area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
#     area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
#     area_union = area_pred + area_lab - area_inter
#     area_sum = area_pred + area_lab
#     assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
#     return area_inter.cpu().numpy(), area_union.cpu().numpy(), area_sum.cpu().numpy()

def batch_pix_accuracy(predict, target, labeled, num_class):
    batch_size = len(predict)
    pixel_correct = torch.zeros((batch_size, num_class))
    #     pixel_labeled = labeled.sum(axis=(-1,-2))
    #     pixel_correct = ((predict == target) * labeled).sum(axis=(-1,-2))
    # # assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    # else:
    for k in range(0, num_class):
        if len(predict.shape) == 3:
            predict_tmp = torch.where(predict == k, True, False)
        else:
            predict_tmp = predict[:,k]

        if len(target.shape) == 3:
            target_tmp = torch.where(target == k, True, False)
        else:
            target_tmp = target[:,k]
        pixel_correct[:,k] = (predict_tmp == target_tmp).sum(axis=(-1,-2))
    return pixel_correct.sum(axis=-1), num_class*labeled.sum(axis=(-1,-2)).cpu()

def batch_intersection_union(predict, target, num_class, verbose=0):
    batch_size = len(predict)
    area_inter = torch.zeros((batch_size, num_class))
    area_union = torch.zeros((batch_size, num_class))
    area_sum = torch.zeros((batch_size, num_class))
    target_sum = torch.zeros((batch_size, num_class))
    for k in range(0, num_class):
        if len(predict.shape) == 3:
            predict_tmp = torch.where(predict == k, True, False)
        else:
            predict_tmp = predict[:,k]

        if len(target.shape) == 3:
            target_tmp = torch.where(target == k, True, False)
        else:
            target_tmp = target[:,k]
        
        area_inter[:,k] = (predict_tmp * target_tmp).sum(axis=(-1,-2))
        area_union[:,k] = (predict_tmp + target_tmp).sum(axis=(-1,-2))
        target_sum_tmp = target_tmp.sum(axis=(-1,-2))
        area_sum[:,k] = predict_tmp.sum(axis=(-1,-2)) + target_sum_tmp
        target_sum[:,k] = target_sum_tmp
        if verbose == 1:
            print('class %d;\n target_num:%s; \n pred_num: %s' %(k, target_tmp.sum(axis=(-1,-2)).cpu(), predict_tmp.sum(axis=(-1,-2)).cpu()))
            print('area_inter: %s;\n area_sum: %s' %(area_inter[:,k], area_sum[:,k]))
    return area_inter, area_union, area_sum, target_sum

def eval_metrics(predict, target, num_class, CoI='all', smooth=0., verbose=0):
    if CoI == 'all':
        CoI = range(num_class)
    batch_size = len(predict)
    if len(target.shape) == 3:
        labeled = torch.ones_like(target)
        # labeled = (target >= 0) * (target < num_class)
    else:
        labeled = torch.ones_like(target[:,0,:,:])
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled, num_class)
    inter, union, union_sum, target_sum = batch_intersection_union(predict, target, num_class, verbose=verbose)
    
    pixAcc = (correct / (num_labeled + 1e-5)).cpu().numpy()
    IoU_all = (inter + smooth) / (union + smooth + 1e-5)
    IoU_db = np.where(union > 0.0, IoU_all, np.nan)
    mIoU = np.nanmean(IoU_db[:,CoI], 1)
    
    Dice_all = (2*inter + smooth) / (union_sum + smooth + 1e-5)
    Dice_db = np.where(union_sum > 0.0, Dice_all, np.nan)
    mDice = np.nanmean(Dice_db[:,CoI], 1)

    # Dice = ( (2.0 * inter + smooth) / (union_sum + smooth)).mean(axis=0).cpu().numpy()
    # mDice = Dice.mean(axis=-1)
    # IoU_all = (inter + smooth / (union + smooth))
    # IoU = (IoU_all.sum(axis=0) / ((target_sum>=1).sum(axis=0) + 1e-5)).cpu().numpy()
    # Dice_all = (2.0 * inter / (union_sum + 1e-5))
    # Dice = (Dice_all.sum(axis=0) / ((target_sum>=1).sum(axis=0) + 1e-5)).cpu().numpy()

    # weight_all = target_sum / (target_sum.sum(axis=-1) + 1e-5).view(batch_size,1)
    # mIoU = (IoU_all*weight_all).sum(axis=-1).mean().cpu().numpy()
    # mDice = (Dice_all*weight_all).sum(axis=-1).mean().cpu().numpy()
    
    return [pixAcc, mIoU, mDice, IoU_db, Dice_db]

    # return {
    #     "Pixel_Accuracy": np.round(pixAcc, 3),
    #     "Mean_IoU": np.round(mIoU, 3),
    #     "Mean_Dice": np.round(mDice, 3),
    #     "Class_IoU": dict(zip(range(num_class), np.round(IoU, 3))),
    #     "Class_Dice": dict(zip(range(num_class), np.round(Dice, 3)))
    # }

# def eval_metrics(output, target, num_class):
#     _, predict = torch.max(output.data, 1)
#     predict = predict + 1
#     target = target + 1

#     labeled = (target > 0) * (target <= num_class)
#     correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
#     inter, union, union_sum = batch_intersection_union(predict, target, num_class, labeled)
#     return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5), np.round(union_sum, 5)]