# pytorch-yolov3
(cocoapi mAP计算在最下方↓↓↓)

# Introduction

This directory contains python software and an iOS App developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. 

# Description

The https://github.com/muyiguangda/pytorch-yolov3 repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.

# Requirements

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.0.0`
- `opencv-python`
- `tqdm`

# Tutorials

* [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
* [Transfer Learning](https://github.com/ultralytics/yolov3/wiki/Example:-Transfer-Learning)
* [Train Single Image](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Image)
* [Train Single Class](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Class)
* [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)

# Training

**Start Training:** Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`.

**Resume Training:** Run `train.py --resume` resumes training from the latest checkpoint `weights/latest.pt`.

Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. Default training settings produce loss plots below, with **training speed of 0.6 s/batch on a 1080 Ti (18 epochs/day)** or 0.45 s/batch on a 2080 Ti.

Here we see training results from `coco_1img.data`, `coco_10img.data` and `coco_100img.data`, 3 example files available in the `data/` folder, which train and test on the first 1, 10 and 100 images of the coco2014 trainval dataset.

`from utils import utils; utils.plot_results()`
![results](https://user-images.githubusercontent.com/26833433/55669383-df76c980-5876-11e9-9806-691bd507ee17.jpg)

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 2 degrees (vertical and horizontal)
Scale | +/- 10%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%


## Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** n1-standard-8 (8 vCPUs, 30 GB memory)  
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.198/hr), P4 ($0.279/hr), T4 ($0.353/hr), P100 ($0.493/hr), V100 ($0.803/hr)  
**HDD:** 100 GB SSD  
**Dataset:** COCO train 2014 

GPUs | `batch_size` | batch time | epoch time | epoch cost
--- |---| --- | --- | --- 
<i></i> |  (images)  | (s/batch) |  |
1 K80 | 16 | 1.43s  | 175min  | $0.58
1 P4 | 8 | 0.51s  | 125min  | $0.58
1 T4 | 16 | 0.78s  | 94min  | $0.55
1 P100 | 16 | 0.39s  | 48min  | $0.39
2 P100 | 32 | 0.48s | 29min | $0.47
4 P100 | 64 | 0.65s | 20min | $0.65
1 V100 | 16 | 0.25s  | 31min | $0.41
2 V100 | 32 | 0.29s | 18min | $0.48
4 V100 | 64 | 0.41s | 13min | $0.70
8 V100 | 128 | 0.49s | 7min | $0.80

# Inference

Run `detect.py` to apply trained weights to an image, such as `zidane.jpg` from the `data/samples` folder:

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights`
<img src="https://user-images.githubusercontent.com/26833433/50524393-b0adc200-0ad5-11e9-9335-4774a1e52374.jpg" width="600">

## Webcam

Run `detect.py` with `webcam=True` to show a live webcam feed.

# Pretrained Weights

- Darknet `*.weights` format: https://pjreddie.com/media/files/yolov3.weights
- PyTorch `*.pt` format: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI

# mAP


**1.下载代码**
sudo rm -rf pytorch-yolov3 && git clone https://github.com/muyiguangda/pytorch-yolov3
**2.获取数据集(可选)**
bash pytorch-yolov3/data/get_coco_dataset.sh
**3.配置cocoapi环境**
cd pytorch-yolov3
sudo rm -rf cocoapi && git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools .
4.**计算mAP**
* Use `python coco_predict.py --weights weights/yolov3.weights` to test the official YOLOv3 weights.
* Use `python coco_predict.py --weights weights/latest.pt` to test the latest training results.
* Use `python coco_predict.py --save-json --conf-thres 0.001 --img-size 416 --batch-size 16` to modify configuration.
* Compare to darknet published results https://arxiv.org/abs/1804.02767.

# pytorch-yolov3
