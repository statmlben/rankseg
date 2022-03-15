import torch

def dice_coeff(pred, target):
    smooth = 0.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum(-1).float()

    return (2. * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth + 1e-5)