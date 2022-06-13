[![dAI](https://img.shields.io/badge/CUHK-dAI-blueviolet)](https://bendai.org)
[![Python](https://badges.aleen42.com/src/python.svg)](https://www.python.org/)
[![Github](https://badges.aleen42.com/src/github.svg)](https://github.com/statmlben/rankseg)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

# 🎲 RankSEG: A Consistent Ranking-based Framework for Segmentation

**RankDice** is a Python module for producing segmentation by `RankDice` framework based on an estimated probability. 

- GitHub repo: [https://github.com/statmlben/rankseg](https://github.com/statmlben/rankseg)
- Documentation: [https://rankseg.readthedocs.io](https://rankseg.readthedocs.io/en/latest/)
- Open Source: [MIT license](https://opensource.org/licenses/MIT)
- Paper: [pdf]()

## Installation

### Dependencies

`rankseg` requires:

| | | | | | |
|-|-|-|-|-|-|
| Python>=3.8 | numpy | torch | sklearn | scipy |

### Source code

You can check the latest sources with the command::

```bash
git clone https://github.com/statmlben/rankseg.git
```

## How-to-Use

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


### Jupyter Notebook
- [Example in VOC 2012 dataset](./pytorch-segmentation/rankdice.ipynb)
- [Simulated examples](./sim/sim.py)

## Illustrative results

### Results in **Fine-annotated Cityscapes dataset**

- `Threshold`, `Argmax` and `rankDice` are performed based on the same network (in `Model` column) trained by the same loss (in `Loss` column). 
- Averaged mDice and mIoU metrics based on state-of-the-art models/losses on **Fine-annotated CityScapes** *val* set. '/' indicates not applicable since the proposed `RankDice`/`mRankDice` requires a strictly proper loss. The best performance in each model/loss is **bold-faced**.

|   | Model       |     | Loss          |     | Threshold (at 0.5)           |     | Argmax                       |     | mRankDice (our)                  |
|---|-------------|-------------|---------------|-------------|------------------------------|-------------|------------------------------|-------------|----------------------------------|
|   |             |             |               |             | (mDice, mIoU) ($\times .01$) |             | (mDice, mIoU) ($\times .01$) |             | (mDice, mIoU) ($\times .01$)     |
|   | DeepLab-V3+ |             | CE            |             | (56.00, 48.40)               |             | (54.20, 46.60)               |             | (**57.80**, **49.80**) |
|   | (resnet101) |             | Focal         |             | (54.10, 46.60)               |             | (53.30, 45.60)               |             | (**56.50**, **48.70**) |
|   |             |             | BCE           |             | (49.80, 24.90)               |             | (44.20, 22.10)               |             | (**54.00**, **27.00**) |
|   |             |             | Soft-Dice     |             | (39.50, 35.90)               |             | (39.50, 35.90)               |             | /                                |
|   |             |             | B-Soft-Dice   |             | (41.00, 20.50)               |             | (27.60, 13.80)               |             | /                                |
|   |             |             | LovaszSoftmax |             | (55.20, 47.60)               |             | (52.30, 45.10)               |             | /                                |
|   | PSPNet      |             | CE            |             | (57.50, 49.60)               |             | (56.50, 48.50)               |             | (**59.30**, **51.00**) |
|   | (resnet50)  |             | Focal         |             | (56.00, 48.20)               |             | (55.80, 47.70)               |             | (**58.20**, **50.00**) |
|   |             |             | BCE           |             | (51.40, 25.70)               |             | (47.60, 23.80)               |             | (**55.10**, **27.60**) |
|   |             |             | Soft-Dice     |             | (49.10, 43.50)               |             | (48.70, 43.20)               |             | /                                |
|   |             |             | B-Soft-Dice   |             | (46.30, 23.10)               |             | (32.70, 16.40)               |             | /                                |
|   |             |             | LovaszSoftmax |             | (56.80, 48.90)               |             | (55.40, 47.70)               |             | /                                |
|   | FCN8        |             | CE            |             | (51.40, 43.70)               |             | (50.50, 42.60)               |             | (**53.50**, **45.30**) |
|   | (resnet101) |             | Focal         |             | (48.50, 41.20)               |             | (49.60, 41.60)               |             | (**51.50**, **43.70**) |
|   |             |             | BCE           |             | (39.40, 19.70)               |             | (39.40, 19.70)               |             | (**41.30**, **20.60**) |
|   |             |             | Soft-Dice     |             | (28.30, 24.30)               |             | (28.30, 24.30)               |             | /                                |
|   |             |             | B-Soft-Dice   |             | (29.10, 14.60)               |             | (29.10, 14.60)               |             | /                                |
|   |             |             | LovaszSoftmax |             | (48.10, 40.40)               |             | (42.90, 35.80)               |             | /                                |

### Results in **PASCAL VOC 2012 dataset**

- `Threshold`, `Argmax` and `rankDice` are performed based on the same network (in `Model` column) trained by the same loss (in `Loss` column). 
- Averaged mDice and mIoU based on state-of-the-art models/losses on **PASCAL VOC 2012** *val* set. '---' indicates that either the performance is significantly worse or the training is unstable, and '/' indicates not applicable since the proposed `RankDice`/`mRankDice` requires a strictly proper loss. The best performance in each model-loss pair is **bold-faced**.

|   | Model       |    | Loss          |    | Threshold (at 0.5)           |    | Argmax                       |    | mRankDice (our)                  |
|---|-------------|-------------|---------------|-------------|------------------------------|-------------|------------------------------|-------------|----------------------------------|
|   |             |             |               |             | (mDice, mIoU) ($\times .01$) |             | (mDice, mIoU) ($\times .01$) |             | (mDice, mIoU) ($\times .01$)     |
|   | DeepLab-V3+ |             | CE            |             | (63.60, 56.70)               |             | (61.90, 55.30)               |             | (**64.01**, **57.01**) |
|   | (resnet101) |             | Focal         |             | (62.70, 55.01)               |             | (60.50, 53.20)               |             | (**62.90**, **55.10**) |
|   |             |             | BCE           |             | (63.30, 31.70)               |             | (59.90, 29.90)               |             | (**64.60**, **32.30**) |
|   |             |             | Soft-Dice     |             | ---                          |             | ---                          |             | /                                |
|   |             |             | B-Soft-Dice   |             | ---                          |             | ---                          |             | /                                |
|   |             |             | LovaszSoftmax |             | (57.70, 51.60)               |             | (56.20, 50.30)               |             | /                                |
|   | PSPNet      |             | CE            |             | (64.60, 57.10)               |             | (63.20, 55.90)               |             | (**65.40**, **57.80**) |
|   | (resnet50)  |             | Focal         |             | (64.00, 56.10)               |             | (63.90, 56.10)               |             | (**66.60**, **58.50**) |
|   |             |             | BCE           |             | (64.20, 32.10)               |             | (65.20, 32.60)               |             | (**67.10, 33.50**)          |
|   |             |             | Soft-Dice     |             | (59.60, 54.00)               |             | (58.80, 53.20)               |             | /                                |
|   |             |             | B-Soft-Dice   |             | (63.30, 31.60)               |             | (54.00. 27.00)               |             | /                                |
|   |             |             | LovaszSoftmax |             | (62.00, 55.20)               |             | (60.80, 54.10)               |             | /                                |
|   | FCN8        |             | CE            |             | (49.50, 41.90)               |             | (45.30, 38.40)               |             | (**50.40**, **42.70**) |
|   | (resnet101) |             | Focal         |             | (50.40, 41.80)               |             | (47.20, 39.30)               |             | (**51.50**, **42.50**) |
|   |             |             | BCE           |             | (46.20, 23.10)               |             | (44.20, 22.10)               |             | (**47.70**, **23.80**) |
|   |             |             | Soft-Dice     |             | ---                          |             | ---                          |             | /                                |
|   |             |             | B-Soft-Dice   |             | ---                          |             | ---                          |             | /                                |
|   |             |             | LovaszSoftmax |             | (39.80, 34.30)               |             | (37.30, 32.20)               |             | /                                |

### More results
- All empirical results on different losses and models can be found [here](./results/test_out.md)

## Replication

If you want to replicate the experiments in our papers, please check the folder `./pytorch-segmentation-rankseg` and its README file [Pytorch-segmentation-rankseg](./pytorch-segmentation-rankseg/README.md)

## To-do list

- [ ] develop a scalable`rank_IoU` with GPU-computing
- [ ] develop a scalable `rank_dice` with non-overlapping segmentation
- [ ] debug for `torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True)` when `enabled=True`
- [ ] CUDA code to speed up the implementation based on `app=1`