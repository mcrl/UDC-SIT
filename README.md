# UDC-SIT: A Real-World Dataset for Under-Display Cameras 

This repository contains the dataset and benchmark DNN models of the following paper. The datasheets for datasets is available in this repository as a pdf file.
> **UDC-SIT: A Real-World Dataset for Under-Display Cameras**<br>
> Kyusu Ahn, Byeonghyun Ko, HyunGyu Lee, Chanwoo Park, and Jaejin Lee<br>
> Under review on Neural Information Processing Systems (**NeurIPS**), 2023<br>

# What is UDC-SIT?
**Well-aligned paired images of Under Display Camera (UDC)** include various environments such as day/night, indoor/outdoor, and flares. The light sources (both natural sunlight and artificial light) and different environmental conditions can lead to various forms of degradation. We have incorporated annotations into our dataset to improve the performance of restoration models for UDC image restoration.


# Why make this?
Under Display Camera (UDC) faces challenges related to image degradation, including issues such as low transmittance, blur, noise, and flare. Despite its significance, there has been a lack of real-world datasets in the UDC domain. Only synthetic images are available that do not accurately represent real-world degradation. As far as we know, it is the first real-world UDC dataset to overcome the problems of the existing UDC datasets.


# Data versions and structure:
You can download our dataset at [our UDC-SIT repository](https://www.dropbox.com/scl/fi/4jtsxjm4xx8q375dt9i9x/UDC-SIT-v2.tar.gz?rlkey=w202pw16w402izohsq2kldpd3&dl=0).
Or, you can download by `wget https://jinpyo.kim/data/UDC-SIT-v2.tar.gz`.

# How can I use this?
You can download the dataset from the link above. When you conduct training, validation, and inference, just normalize in your PyTorch DataLoader as is generally being done in most image restoration DNN models. We recommend to train your model using `.npy` format with 4 channels rather than converting it to a 3 channels RGB domain. Note that our dataset is in Low Dynamic Range (LDR). Therefore, you don't have to conduct tone-mapping. You can visualize our dataset by running `./dataset/visualize_sit.py`, allowing for a visual inspection.


# Annotation details
- Filename: Filename in .npy format
- Dataset: Classification of the data into training/validation/test sets
- Indoor/Outdoor: Photos taken indoors (1) or outdoors (3)
- Day/Night: Classification of photos as not requiring day/night differentiation indoors (1), taken during the day (2), or taken at night (3)
- Glare: Presence (1) or absence (0) of glare in the photos
- Shimmer: Presence (1) or absence (0) of shimmer in the photos
- Streak: Presence (1) or absence (0) of streaks in the photos
- Light source: Absence of a flare-inducing light source (0), natural sunlight (1), artificial light (2), or a combination of natural sunlight and artificial light (3)


# Who created this dataset?
The dataset is created by the authors of the paper as well as the members of the Thunder Research Group at Seoul National University, including Woojin Kim, Gyuseong Lee, Dongyoung Lee, Sangsoo Im, Gwangho Choi, Gyeongje Jo, Yeonkyoung So, Jiheon Seok, Jaehwan Lee, Donghun Choi, and Daeyoung Park, on behalf of universities and research institutions.


# Licences 
Copyright (c) 2023 Thunder Research Group

All software for benchmark Deep Neural Network (DNN) models adheres to the license of the original authors. You can find the original source codes and their respective licenses for ECFNet, UDC-UNet, DISCNet, and Uformer in the links below.
- ECFNet (https://github.com/zhuyr97/ECFNet)
- UDC-UNet (https://github.com/J-FHu/UDCUNet)
- DISCNet (https://github.com/jnjaby/DISCNet)
- Uformer (https://github.com/ZhendongWang6/Uformer)

In addition, UDC-SIT dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). This means that you are allowed to freely utilize, share, and modify this work under the condition of properly attributing the original author, distributing any derived works under the same license, and utilizing it exclusively for non-commercial purposes.
