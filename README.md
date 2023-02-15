[![dAI](https://img.shields.io/badge/CUHK-dAI-blueviolet)](https://bendai.org)
[![Python](https://badges.aleen42.com/src/python.svg)](https://www.python.org/)
[![Github](https://badges.aleen42.com/src/github.svg)](https://github.com/statmlben/rankseg)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/statmlben/rankseg/pulls)

# 🎲 RankSEG: A Consistent Ranking-based Framework for Segmentation

**RankDice** is a Python module for producing segmentation by `RankDice` framework based on an estimated probability. 

- GitHub repo: [https://github.com/statmlben/rankseg](https://github.com/statmlben/rankseg)
- Open Source: [MIT license](https://opensource.org/licenses/MIT)
- Paper: [arXiv:2206.13086](https://arxiv.org/abs/2206.13086)
<!-- - Documentation: [https://rankseg.readthedocs.io](https://rankseg.readthedocs.io/en/latest/) -->

## Installation

### Dependencies

`rankseg` requires **Python 3.9** + [Python libraries](./requirements.txt):

```bash
pip install -r requirements.txt
```
### Source code

You can check the latest sources with the command::

```bash
git clone https://github.com/statmlben/rankseg.git
```

## How-to-Use (*on a batch*)

### RankDice
```python
## `out_prob` (batch_size, num_class, width, height) is the output probability for each pixel based on a trained neural network
from rankseg import rank_dice
predict_rd, tau_rd, cutpoint_rd = rank_dice(out_prob, app=2, device='cuda')
```
### Other existing frameworks (`Threshold` and `Argmax`)
```python
## `out_prob` (batch_size, num_class, width, height) is the output probability for each pixel based on a trained neural network

## Threshold
predict_T = torch.where(out_prob > .5, True, False)

## Argmax
idx = torch.argmax(out_prob.data, dim=1, keepdims=True)
predict_max = torch.zeros_like(out_prob.data, dtype=bool).scatter_(1, idx, True)
```

### Usage in `pytorch-segmentation-rankseg` (in subfolder)
```bash
## rankdice
$ python test.py -r saved/cityscapes/PSPNet/CrossEntropyLoss2d/T/05-04_13-08/checkpoint-epoch300.pth -p "rankdice"

TEST, Pred (rankdice) | Loss: 0.159, PixelAcc: 0.99, Mean IoU: 0.51, Mean Dice 0.59 |: 100%|██████| 84/84 [01:03<00:00,  1.33it/s]

    ## TESTING Restuls for Model: PSPNet + Loss: CrossEntropyLoss2d + predict: rankdice ## 
         test_loss      : 0.15925
         Pixel_Accuracy : 0.9879999756813049
         Mean_IoU       : 0.5099999904632568
         Mean_Dice      : 0.5929999947547913
         Class_IoU      : {0: 0.771, 1: 0.508, 2: 0.767, 3: 0.164, 4: 0.117, 5: 0.317, 6: 0.283, 7: 0.401, 8: 0.841, 9: 0.231, 10: 0.778, 11: 0.4, 12: 0.292, 13: 0.766, 14: 0.233, 15: 0.465, 16: 0.315, 17: 0.177, 18: 0.326}
         Class_Dice     : {0: 0.856, 1: 0.608, 2: 0.851, 3: 0.21, 4: 0.158, 5: 0.46, 6: 0.374, 7: 0.514, 8: 0.903, 9: 0.294, 10: 0.845, 11: 0.495, 12: 0.372, 13: 0.84, 14: 0.265, 15: 0.513, 16: 0.358, 17: 0.222, 18: 0.419}

## max
$ python test.py -r saved/cityscapes/PSPNet/CrossEntropyLoss2d/T/05-04_13-08/checkpoint-epoch300.pth -p "max"

TEST, Pred (max) | Loss: 0.159, PixelAcc: 0.99, Mean IoU: 0.49, Mean Dice 0.56 |: 100%|███████████| 84/84 [00:12<00:00,  6.52it/s]

    ## TESTING Restuls for Model: PSPNet + Loss: CrossEntropyLoss2d + predict: max ## 
         test_loss      : 0.15925
         Pixel_Accuracy : 0.9879999756813049
         Mean_IoU       : 0.48500001430511475
         Mean_Dice      : 0.5649999976158142
         Class_IoU      : {0: 0.768, 1: 0.489, 2: 0.759, 3: 0.133, 4: 0.099, 5: 0.295, 6: 0.257, 7: 0.387, 8: 0.836, 9: 0.208, 10: 0.769, 11: 0.372, 12: 0.272, 13: 0.751, 14: 0.204, 15: 0.395, 16: 0.268, 17: 0.152, 18: 0.303}
         Class_Dice     : {0: 0.854, 1: 0.585, 2: 0.844, 3: 0.172, 4: 0.136, 5: 0.428, 6: 0.341, 7: 0.498, 8: 0.9, 9: 0.268, 10: 0.835, 11: 0.464, 12: 0.351, 13: 0.826, 14: 0.233, 15: 0.437, 16: 0.308, 17: 0.193, 18: 0.392}


## threshold at 0.5
$ python test.py -r saved/cityscapes/PSPNet/CrossEntropyLoss2d/T/05-04_13-08/checkpoint-epoch300.pth -p "T"

TEST, Pred (T) | Loss: 0.159, PixelAcc: 0.99, Mean IoU: 0.50, Mean Dice 0.57 |: 100%|█████████████| 84/84 [00:13<00:00,  6.45it/s]

    ## TESTING Restuls for Model: PSPNet + Loss: CrossEntropyLoss2d + predict: T ## 
         test_loss      : 0.15925
         Pixel_Accuracy : 0.9890000224113464
         Mean_IoU       : 0.4959999918937683
         Mean_Dice      : 0.574999988079071
         Class_IoU      : {0: 0.772, 1: 0.478, 2: 0.762, 3: 0.136, 4: 0.109, 5: 0.29, 6: 0.265, 7: 0.39, 8: 0.841, 9: 0.201, 10: 0.77, 11: 0.363, 12: 0.273, 13: 0.769, 14: 0.219, 15: 0.422, 16: 0.307, 17: 0.158, 18: 0.325}
         Class_Dice     : {0: 0.857, 1: 0.573, 2: 0.846, 3: 0.174, 4: 0.147, 5: 0.419, 6: 0.349, 7: 0.499, 8: 0.902, 9: 0.257, 10: 0.836, 11: 0.451, 12: 0.351, 13: 0.841, 14: 0.247, 15: 0.468, 16: 0.349, 17: 0.197, 18: 0.414}
```

### Jupyter Notebook
- [Example in VOC 2012 dataset](./rankdice.ipynb)
- [Simulated examples](./sim/sim.py)

## Illustrative results

### Results in **Fine-annotated Cityscapes dataset**

- `Threshold`, `Argmax` and `rankDice` are performed based on the same network (in `Model` column) trained by the same loss (in `Loss` column). 
- Averaged mDice and mIoU metrics based on state-of-the-art models/losses on **Fine-annotated CityScapes** *val* set. '/' indicates not applicable since the proposed `RankDice`/`mRankDice` requires a strictly proper loss. The best performance in each model/loss is **bold-faced**.
- All trained neural networks and their `config.json` with different `network` and `loss` are saved in [this link](https://gocuhk-my.sharepoint.com/:f:/g/personal/bendai_cuhk_edu_hk/EmFAT0kVqBhBoWg-5JMLwK4BK3SugqMXL5GoRpVT6gG4xg?e=RQvUpz) (**12G** folder: network/loss/.../`*.pth` + `config.json`)

| Model       | Loss          | Threshold (at 0.5)          | Argmax                       | mRankDice (our)              |
|-------------|---------------|-----------------------------|------------------------------|------------------------------|
|             |               | (mDice, mIoU) ($\times .01$) | (mDice, mIoU) ($\times .01$) | (mDice, mIoU) ($\times .01$) |
| DeepLab-V3+ | CE            | (56.00, 48.40)              | (54.20, 46.60)               | (**57.80**, **49.80**)       |
| (resnet101) | Focal         | (54.10, 46.60)              | (53.30, 45.60)               | (**56.50**, **48.70**)       |
|             | BCE           | (49.80, 24.90)              | (44.20, 22.10)               | (**54.00**, **27.00**)       |
|             | Soft-Dice     | (39.50, 35.90)              | (39.50, 35.90)               | /                            |
|             | B-Soft-Dice   | (41.00, 20.50)              | (27.60, 13.80)               | /                            |
|             | LovaszSoftmax | (55.20, 47.60)              | (52.30, 45.10)               | /                            |
| PSPNet      | CE            | (57.50, 49.60)              | (56.50, 48.50)               | (**59.30**, **51.00**)       |
| (resnet50)  | Focal         | (56.00, 48.20)              | (55.80, 47.70)               | (**58.20**, **50.00**)       |
|             | BCE           | (51.40, 25.70)              | (47.60, 23.80)               | (**55.10**, **27.60**)       |
|             | Soft-Dice     | (49.10, 43.50)              | (48.70, 43.20)               | /                            |
|             | B-Soft-Dice   | (46.30, 23.10)              | (32.70, 16.40)               | /                            |
|             | LovaszSoftmax | (56.80, 48.90)              | (55.40, 47.70)               | /                            |
| FCN8        | CE            | (51.40, 43.70)              | (50.50, 42.60)               | (**53.50**, **45.30**)       |
| (resnet101) | Focal         | (48.50, 41.20)              | (49.60, 41.60)               | (**51.50**, **43.70**)       |
|             | BCE           | (39.40, 19.70)              | (39.40, 19.70)               | (**41.30**, **20.60**)       |
|             | Soft-Dice     | (28.30, 24.30)              | (28.30, 24.30)               | /                            |
|             | B-Soft-Dice   | (29.10, 14.60)              | (29.10, 14.60)               | /                            |
|             | LovaszSoftmax | (48.10, 40.40)              | (42.90, 35.80)               | /                            |


### Results in **PASCAL VOC 2012 dataset**

- `Threshold`, `Argmax` and `rankDice` are performed based on the same network (in `Model` column) trained by the same loss (in `Loss` column). 
- Averaged mDice and mIoU based on state-of-the-art models/losses on **PASCAL VOC 2012** *val* set. '---' indicates that either the performance is significantly worse or the training is unstable, and '/' indicates not applicable since the proposed `RankDice`/`mRankDice` requires a strictly proper loss. The best performance in each model-loss pair is **bold-faced**.
- All trained neural networks with different `network` and `loss` are saved in [this link](https://gocuhk-my.sharepoint.com/:f:/g/personal/bendai_cuhk_edu_hk/EkqD1EH7bBVImHcWowJ7jR8BfatVPsOFFGkSsfvMjm0juQ?e=6LO8vI) (**22G** folder: network/loss/.../*.pth)

| Model       | Loss          | Threshold (at 0.5)          | Argmax                       | mRankDice (our)              |
|-------------|---------------|-----------------------------|------------------------------|------------------------------|
|             |               | (mDice, mIoU) ($\times .01$) | (mDice, mIoU) ($\times .01$) | (mDice, mIoU) ($\times .01$) |
| DeepLab-V3+ | CE            | (63.60, 56.70)              | (61.90, 55.30)               | (**64.01**, **57.01**)       |
| (resnet101) | Focal         | (62.70, 55.01)              | (60.50, 53.20)               | (**62.90**, **55.10**)       |
|             | BCE           | (63.30, 31.70)              | (59.90, 29.90)               | (**64.60**, **32.30**)       |
|             | Soft-Dice     | ---                         | ---                          | /                            |
|             | B-Soft-Dice   | ---                         | ---                          | /                            |
|             | LovaszSoftmax | (57.70, 51.60)              | (56.20, 50.30)               | /                            |
| PSPNet      | CE            | (64.60, 57.10)              | (63.20, 55.90)               | (**65.40**, **57.80**)       |
| (resnet50)  | Focal         | (64.00, 56.10)              | (63.90, 56.10)               | (**66.60**, **58.50**)       |
|             | BCE           | (64.20, 32.10)              | (65.20, 32.60)               | (**67.10, 33.50**)           |
|             | Soft-Dice     | (59.60, 54.00)              | (58.80, 53.20)               | /                            |
|             | B-Soft-Dice   | (63.30, 31.60)              | (54.00. 27.00)               | /                            |
|             | LovaszSoftmax | (62.00, 55.20)              | (60.80, 54.10)               | /                            |
| FCN8        | CE            | (49.50, 41.90)              | (45.30, 38.40)               | (**50.40**, **42.70**)       |
| (resnet101) | Focal         | (50.40, 41.80)              | (47.20, 39.30)               | (**51.50**, **42.50**)       |
|             | BCE           | (46.20, 23.10)              | (44.20, 22.10)               | (**47.70**, **23.80**)       |
|             | Soft-Dice     | ---                         | ---                          | /                            |
|             | B-Soft-Dice   | ---                         | ---                          | /                            |
|             | LovaszSoftmax | (39.80, 34.30)              | (37.30, 32.20)               | /                            |

### Results in **Kvasir-SEG dataset**

- `Threshold`, `Argmax` and `rankDice` are performed based on the same network (in `Model` column) trained by the same loss (in `Loss` column). 
- `Threshold` and `Argmax` are exactly the same in **binary segmentation**. 
- Averaged mDice and mIoU based on state-of-the-art models/losses on **Kvasir-SEG dataset** set. '---' indicates that either the performance is significantly worse or the training is unstable, and '/' indicates not applicable since the proposed `RankDice`/`mRankDice` requires a strictly proper loss. The best performance in each model-loss pair is **bold-faced**.
<!-- - All trained neural networks with different `network` and `loss` are saved in [this link](https://gocuhk-my.sharepoint.com/:f:/g/personal/bendai_cuhk_edu_hk/EkqD1EH7bBVImHcWowJ7jR8BfatVPsOFFGkSsfvMjm0juQ?e=6LO8vI) (**22G** folder: network/loss/.../*.pth) -->

| Model       | Loss          | Threshold/Argmax           | mRankDice (our)                |
|-------------|---------------|----------------------------|--------------------------------|
|             |               | (Dice, IoU) ($\times .01$) | (Dice, IoU) ($\times .01$)     |
| DeepLab-V3+ | CE            | (87.9, 80.7)               | **(88.3, 80.9)**               |
| (resnet101) | Focal         | (86.5, 87.3)               | /                              |
|             | Soft-Dice     | (85.7, 77.8)               | /                              |
|             | LovaszSoftmax | (84.3, 77.3)               | /                              |
| PSPNet      | CE            | (86.3, 79.2)               | **(87.1, 79.8)**               |
| (resnet50)  | Focal         | (83.8, 75.4)               | /                              |
|             | Soft-Dice     | (83.5, 75.9)               | /                              |
|             | LovaszSoftmax | (86.0, 79.2)               | /                              |
| FCN8        | CE            | (81.9, 73.5)               | **(82.1, 73.6)**               |
| (resnet101) | Focal         | (78.5, 69.0)               | /                              |
|             | Soft-Dice     | ---                        | ---                            |
|             | LovaszSoftmax | (82.0, 73.4)               | /                              |

### More results
- All empirical results on different losses and models can be found [here](./results/test_out.md)

## Replication

If you want to replicate the experiments in our papers, please check the folder `./pytorch-segmentation-rankseg` and its README file [Pytorch-segmentation-rankseg](./pytorch-segmentation-rankseg/README.md)


## Citation
If you use `RankSEG` for an academic publication, we would appreciate citations to the following paper:

```
@misc{dai2022rankseg,
    doi = {10.48550/ARXIV.2206.13086},
    url = {https://arxiv.org/abs/2206.13086},
    author = {Dai, Ben and Li, Chunlin},
    title = {RankSEG: A Consistent Ranking-based Framework for Segmentation},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

## To-do list 

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/statmlben/rankseg/pulls)
[![Github](https://badges.aleen42.com/src/github.svg)](https://github.com/statmlben/rankseg)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- [ ] develop `rank_dice` for `numpy` and `tf2` 
- [ ] develop a scalable`rank_IoU` with GPU-computing
- [ ] develop a scalable `rank_dice` with non-overlapping segmentation
- [ ] debug for `torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True)` when `enabled=True`
- [ ] CUDA code to speed up the implementation based on `app=1`

