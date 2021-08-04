## Face Mask Detection 

This is a face mask detection project using **Faster R-CNN-R50** model based on [Detectron2](https://github.com/facebookresearch/detectron2).

## Installation

#### Requirements

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps

1. Install and build libs

```
git clone https://github.com/JiaminWoo/Face_Mask_Detection.git
Face_Mask_Detection
python setup.py build develop
```

2. Prepare datasets. Our dataset is downloaded from [AIZOOTech](https://github.com/AIZOOTech/FaceMaskDetection). And we convert it into Pascal Voc format for convenience. Then we split the data into train and val data set. We also correct some wrong label in the original data set. The final processed data is uploaded on [Google drive]().
3. Link dataset path to Face_Mask_Detection/datasets/facemask

```
ln -s /path_to_facemask_dataset/facemask datasets/facemask
```

## Getting Started

1. Train our model. You can tune the number of GPUs using `--num-gpus` per you need.

```
python tools/train_net.py --num-gpus 4 \
    --config-file projects/MaskDet/configs/facemask.yaml
```

2. Evaluate our model

```
python tools/train_net.py --num-gpus 4 \
    --config-file projects/MaskDet/configs/facemask.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

3. Infer our model by image

```
python demo/demo.py\
    --config-file projects/MaskDet/configs/facemask.yaml \
    --input path/to/images --output path/to/save_images --confidence-threshold 0.4 \
    --opts MODEL.WEIGHTS path/to/model.pth
```

4. Infer our model by video

```
python demo/demo.py\
    --config-file projects/MaskDet/configs/facemask.yaml \
    --video-input path/to/video --output path/to/save_video_file \
    --opts MODEL.WEIGHTS path/to/model.pth
```

Learn more at Detectron2's [documentation](https://detectron2.readthedocs.org).

## Pre-trained Model

We also provide our trained model on [Google drive]() so that you can directly download it to make demo.
