# Semantic Segmentation in PyTorch (`RankDice` included)

This repo is forked from [yassouali/pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation) with the following updates:

- [x] `mDice` and `mIoU` are computed based on Appendix A in [our paper](https://arxiv.org/pdf/2206.13086.pdf)
- [x] More segmentation losses including `BCEWithLogitsLoss2d`, `BCEWithLogitsFocalLoss2d`, `BDiceLoss`, and `LogCoshDiceLoss` are added.
- [x] Three procedures (`T`, `max`, `rankdice`) are included, where `T` indicates thresholding at 0.5, `max` indicates taking argmax over classes, `rankdice` indicates algo in Algorithm 2 in [our paper](https://arxiv.org/pdf/2206.13086.pdf). You can use one of them in train/val/test by setting `config.json`
- [x] Class of interest `CoI` to exclude some classes in testing, yet you still want to keep them in training. 
    ```json
        "predict": {"CoI": [0,1],
                    "train": "T",
                    "val": "max",
                    "test": "rankdice"}
    ``` 
- [x] `test.py` to report the score for a testing set, with three options to produce segmentation: `rankdice`, `max` and `T`:
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


Following is the original README file from [yassouali/pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation)

---
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<!-- TOC -->

- Semantic Segmentation in PyTorch (RankDice included)
  - [Requirements](#requirements)
  - [Main Features](#main-features)
    - [Models](#models)
    - [Datasets](#datasets)
    - [Losses](#losses)
    - [Learning rate schedulers](#learning-rate-schedulers)
    - [Data augmentation](#data-augmentation)
  - [Training](#training)
  - [Inference](#inference)
  - [Code structure](#code-structure)
  - [Config file format](#config-file-format)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

This repo contains a PyTorch an implementation of different semantic segmentation models for different datasets.

## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.

```bash
pip install -r requirements.txt
```

or for a local installation

```bash
pip install --user -r requirements.txt
```

## Main Features

- A clear and easy to navigate structure,
- A `json` config file with a lot of possibilities for parameter tuning,
- Supports various models, losses, Lr schedulers, data augmentations and datasets,

**So, what's available ?**

### Models 
- (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611)
- (**GCN**) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [[Paper]](https://arxiv.org/abs/1703.02719)
- (**UperNet**) Unified Perceptual Parsing for Scene Understanding [[Paper]](https://arxiv.org/abs/1807.10221)
- (**DUC, HDC**) Understanding Convolution for Semantic Segmentation [[Paper]](https://arxiv.org/abs/1702.08502) 
- (**PSPNet**) Pyramid Scene Parsing Network [[Paper]](http://jiaya.me/papers/PSPNet_cvpr17.pdf) 
- (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[Paper]](https://arxiv.org/abs/1606.02147)
- (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper]](https://arxiv.org/abs/1505.04597)
- (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [[Paper]](https://arxiv.org/pdf/1511.00561)
- (**FCN**) Fully Convolutional Networks for Semantic Segmentation (2015): [[Paper]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 

### Datasets

- **Pascal VOC:** For pascal voc, first download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` containing, the image sets, the XML annotation for both object detection and segmentation, and JPEG images.\
The second step is to augment the dataset using the additionnal annotations provided by [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf). First download the image sets (`train_aug`, `trainval_aug`, `val_aug` and `test_aug`) from this link: [Aug ImageSets](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=), and  add them the rest of the segmentation sets in `/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation`, and then download new annotations [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and add them to the path `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`, now we're set, for training use the path to `VOCtrainval_11-May-2012`

- **CityScapes:** First download the images and the annotations (there is two types of annotations, Fine `gtFine_trainvaltest.zip` and Coarse `gtCoarse.zip` annotations, and the images `leftImg8bit_trainvaltest.zip`) from the official website [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/), extract all of them in the same folder, and use the location of this folder in `config.json` for training.

- **ADE20K:** For ADE20K, simply download the images and their annotations for training and validation from [sceneparsing.csail.mit.edu](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), and for the rest visit the [website](http://sceneparsing.csail.mit.edu/).


- **COCO Stuff:** For COCO, there is two partitions, CocoStuff10k with only 10k that are used for training the evaluation, note that this dataset is outdated, can be used for small scale testing and training, and can be downloaded [here](https://github.com/nightrome/cocostuff10k). For the official dataset with all of the training 164k examples, it can be downloaded from the official [website](http://cocodataset.org/#download).\
Note that when using COCO dataset, 164k version is used per default, if 10k is prefered, this needs to be specified with an additionnal parameter `partition = 'CocoStuff164k'` in the config file with the corresponding path.

### Losses
In addition to the Cross-Entorpy loss, there is also
- **Dice-Loss**, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize.
- **CE Dice loss**, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.
- **Focal Loss**, an alternative version of the CE, used to avoid class imbalance where the confident predictions are scaled down.
- **Lovasz Softmax** lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses (for more details, check the paper: [The Lovász-Softmax loss](https://arxiv.org/abs/1705.08790)).

### Learning rate schedulers
- **Poly learning rate**, where the learning rate is scaled down linearly from the starting value down to zero during training. Considered as the go to scheduler for semantic segmentaion (see Figure below).
- **One Cycle learning rate**, for a learning rate LR, we start from LR / 10 up to LR for 30% of the training time, and we scale down to LR / 25 for remaining time, the scaling is done in a cos annealing fashion (see Figure bellow), the momentum is also modified but in the opposite manner starting from 0.95 down to 0.85 and up to 0.95, for more detail see the paper: [Super-Convergence](https://arxiv.org/abs/1708.07120). 

<p align="center"><img src="images/learning_rates.png" align="center" width="750"></p>

### Data augmentation
All of the data augmentations are implemented using OpenCV in `\base\base_dataset.py`, which are: rotation (between -10 and 10 degrees), random croping between 0.5 and 2 of the selected `crop_size`, random h-flip and blurring

## Training
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:

```bash
python train.py --config config.json
```

The training will automatically be run on the GPUs (if more that one is detected and  multipple GPUs were selected in the config file, `torch.nn.DataParalled` is used for multi-gpu training), if not the CPU is used. The log files will be saved in `saved\runs` and the `.pth` chekpoints in `saved\`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```

<p align="center"><img src="images/tb1.png" align="center" width="900"></p>

<p align="center"><img src="images/tb2.png" align="center" width="900"></p>

## Inference

For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --config config.json --model best_model.pth --images images_folder
```

The predictions will be saved as `.png` images using the default palette in the passed fodler name, if not, `outputs\` is used, for Pacal VOC the default palette is:

<p align="center"><img src="images/colour_scheme.png" align="center" width="550"></p>

Here are the parameters availble for inference:
```
--output       The folder where the results will be saved (default: outputs).
--extension    The extension of the images to segment (default: jpg).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--mode         Mode to be used, choose either `multiscale` or `sliding` for inference (multiscale is the default behaviour).
--config       The config file used for training the model.
```

**Trained Model:**

| Model     | Backbone     | PascalVoc val mIoU | PascalVoc test mIoU | Pretrained Model |
| :-------- | :----------: |:-----------------: |:-------------------:|:----------------:|
| PSPNet    | ResNet 50    | 82%                | 79%                 | [Dropbox](https://www.dropbox.com/s/7rni1tzyjmdkiup/PSPnet.pth?dl=0) |



## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── inference.py - inference using a trained model
  ├── trainer.py - the main trained
  ├── config.json - holds configuration for training
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```

## Config file format
Config files are in `.json` format:
```javascript
{
  "name": "PSPNet",         // training session name
  "n_gpu": 1,               // number of GPUs to use for training.
  "use_synch_bn": true,     // Using Synchronized batchnorm (for multi-GPU usage)

    "arch": {
        "type": "PSPNet", // name of model architecture to train
        "args": {
            "backbone": "resnet50",     // encoder type type
            "freeze_bn": false,         // When fine tuning the model this can be used
            "freeze_backbone": false    // In this case only the decoder is trained
        }
    },

    "train_loader": {
        "type": "VOC",          // Selecting data loader
        "args":{
            "data_dir": "data/",  // dataset path
            "batch_size": 32,     // batch size
            "augment": true,      // Use data augmentation
            "crop_size": 380,     // Size of the random crop after rescaling
            "shuffle": true,
            "base_size": 400,     // The image is resized to base_size, then randomly croped
            "scale": true,        // Random rescaling between 0.5 and 2 before croping
            "flip": true,         // Random H-FLip
            "rotate": true,       // Random rotation between 10 and -10 degrees
            "blur": true,         // Adding a slight amount of blut to the image
            "split": "train_aug", // Split to use, depend of the dataset
            "num_workers": 8
        }
    },

    "val_loader": {     // Same for val, but no data augmentation, only a center crop
        "type": "VOC",
        "args":{
            "data_dir": "data/",
            "batch_size": 32,
            "crop_size": 480,
            "val": true,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,      // Using lr/10 for the backbone, and lr for the rest
        "args":{
            "lr": 0.01,               // Learning rate
            "weight_decay": 1e-4,     // Weight decay
            "momentum": 0.9
        }
    },

    "loss": "CrossEntropyLoss2d",     // Loss (see utils/losses.py)
    "ignore_index": 255,              // Class to ignore (must be set to -1 for ADE20K) dataset
    "lr_scheduler": {   
        "type": "Poly",               // Learning rate scheduler (Poly or OneCycle)
        "args": {}
    },

    "trainer": {
        "epochs": 80,                 // Number of training epochs
        "save_dir": "saved/",         // Checkpoints are saved in save_dir/models/
        "save_period": 10,            // Saving chechpoint each 10 epochs
  
        "monitor": "max Mean_IoU",    // Mode and metric for model performance 
        "early_stop": 10,             // Number of epochs to wait before early stoping (0 to disable)
        
        "tensorboard": true,        // Enable tensorboard visualization
        "log_dir": "saved/runs",
        "log_per_iter": 20,         

        "val": true,
        "val_per_epochs": 5         // Run validation each 5 epochs
    }
}
```



## Acknowledgement
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m)
- [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
