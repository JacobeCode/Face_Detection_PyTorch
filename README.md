# Face_Detection_PyTorch

## Current state

Main frame and model prepared - currently training (2 full runs on basic params).

## Target of repository

Currently this repository will be used for creating live object identification. It will be done on animal-like database or face emotions dataset.

Ultimately I will try to fine-tune existing models, but it is possible that I will try create some with use of PyTorch or Keras.

### Used Database - [Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

With thanks for possibility to use mentioned database:

*StarGAN v2: Diverse Image Synthesis for Multiple Domains, Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020*

### Content

In this repository you can find files:
- data_processing - all the files conected to data manipulation
    - transforms.py - transforms created for use on the dataset images e.g. rotation
- detect_analysis.ipynb - Jupyter Notebook used for data analysis and manipulation
- real_time.py - a simple implementation of live object detection with use of camera (currently with use of YOLO)

### Approach Description

#1 Classic architecture with database padding to same maximal size

### Additional Content

Database:
> choi2020starganv2,
> title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
> author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
> booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
> year={2020}

YOLO:
> Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
