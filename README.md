[![dAI](https://img.shields.io/badge/CUHK-dAI-blueviolet)](https://bendai.org)
[![Python](https://badges.aleen42.com/src/python.svg)](https://www.python.org/)
[![Github](https://badges.aleen42.com/src/github.svg)](https://github.com/statmlben/rankseg)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

# 🎲 RankSEG: A Consistent Ranking-based Framework for Segmentation

**RankSEG** is a Python module for producing segmentation based on an estimated probability. 

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

## Example

- Notebook 1: [Simulated example](./sim.py)
- Notebook 2: [`Pytorch-segmentation` in VOC 2012 dataset](./pytorch-segmentation/rankdice.ipynb)

## Replicate an experiment in the paper

- README: [Pytorch-segmentation](./pytorch-segmentation/README.md)
