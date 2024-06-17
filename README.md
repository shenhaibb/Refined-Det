# Refined-Det

This repository contains the Pytorch implementations of the following paper:

Xiaohan Rao, and Liming Zhou. Cross-Grid Label Assignment for Arbitrary-Oriented Object Detection in Aerial Images. &nbsp; [IEEE GRSL, 2024](https://ieeexplore.ieee.org/document/10546292)

<!-- [ALGORITHM] -->

## Abstract

As a challenging task in the field of remote sensing, object detection has attracted widespread attention from researchers. However, for aerial images with an imbalanced foreground–background distribution, the existing label assignment assigns insufficient positive samples to aerial objects, severely limiting detection performance. In this letter, we propose the cross-grid label assignment (CLA) to add high-quality positive samples used for training and loss calculation, thereby alleviating the issue of imbalanced positive and negative samples. Furthermore, the feature refinement head (FRHead), which extracts object-oriented features and guiding semantic enhancement, is used to address the inconsistent between classification scores and localization accuracy. Extensive experiments have shown that our method has superior detection performance, with 90.50% and 73.69% mAP on the HRSC2016 and DOTA datasets, respectively.

<p align="center"> <img src="https://raw.github.com/shenhaibb/Refined-Det/main/imgs/Overall Network Architecture.jpg" alt="CLA"></p>

## Installation

Our model is based on [MMRotate](https://github.com/open-mmlab/mmrotate), [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n mmrotate python=3.7
conda activate mmrotate
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmengine==0.6.0
mim install mmcv-full==1.7.0
mim install mmdet==2.28.0
pip install -r requirements/build.txt
pip install -v -e .
```

## Citation

```
@ARTICLE{10546292,
  author={Rao, Xiaohan and Zhou, Liming},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Cross-Grid Label Assignment for Arbitrary-Oriented Object Detection in Aerial Images}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Feature extraction;Training;Kernel;Object detection;Head;Shape;Semantics;Aerial images;anchor-based detector;feature alignment;label assignment;oriented object detection},
  doi={10.1109/LGRS.2024.3408148}}
```

## Contact

**Any question regarding this work can be addressed to [rshenhaibb@stu.hit.edu.cn](rshenhaibb@stu.hit.edu.cn).**