# Under Display Camera dataset for Still Image by Thunder Research Group (UDC-SIT)

This is the companion code repository for [UDC-SIT] and its article, used to generate the dataset and train several restoration benchmarks on it. The associated datasheet for dataset is available in this repository as a pdf file.

# What is UDC-SIT?
**Well-aligned paired images of Under Display Camera (UDC)** include various environments such as day/night, indoor/outdoor, and flares. The light sources (both natural sunlight and artificial light) and different environmental conditions can lead to various forms of degradation. We have incorporated annotations into our dataset to improve the performance of restoration models for UDC image restoration.


# Why make this?
UDC is a technology that integrates a camera module beneath the display, allowing the UDC area to function as a regular display while capturing images when the camera is in operation. However, UDC faces challenges related to image deterioration, including issues such as low transmittance, blurriness, noise, and flare. Consequently, UDC image restoration becomes a crucial and challenging task. Despite its significance, there has been a lack of real-world datasets in this domain, with only synthetic images available that do not accurately represent the real degradation. Our aim is to contribute to active research and advancement in UDC image restoration by providing a well-aligned real-world dataset.


# Data versions and structure:
You can download our dataset at [UDC-SIT](https://www.dropbox.com/s/9j0v2nnvr0eo91g/UDC-SIT.tar.gz?dl=0).


# How can I use this?
You can download the dataset from the repository. We recommend starting with running `./dataset/visualize_sit.py` to explore the data. This script visualizes our dataset by converting it into PNG format, allowing for a visual inspection. When you conduct training, validation, and inference, just normalize in your DataLoader as is generally being done in most image restoration models. We recommend to train your model using `.npy` format with 4 channels we provide rather than converting it to a PNG file. Note that our dataset is in Low Dynamic Range (LDR). Therefore, you don't have to conduct tone-mapping.


# Annotation details
- Filename: Filename in .npy format
- Dataset: Classification of the data into training/validation/test sets
- Indoor/Outdoor: Photos taken indoors (1) or outdoors (3)
- Day/Night: Classification of photos as not requiring day/night differentiation indoors (1), taken during the day (2), or taken at night (3)
- Glare: Presence (1) or absence (0) of glare in the photos
- Shimmer: Presence (1) or absence (0) of shimmer in the photos
- Streak: Presence (1) or absence (0) of streaks in the photos
- Light source: Absence of a flare-inducing light source (0), natural sunlight (1), artificial light (2), or a combination of natural sunlight and artificial light (3)


# Licences 
Copyright (c) 2023 Thunder Research Group

All software for benchmark Deep Neural Network (DNN) models adheres to the license of the original authors. You can find the original source codes and their respective licenses for ECFNet, UDC-UNet, DISCNet, and Uformer in the links below.
- ECFNet (https://github.com/zhuyr97/ECFNet)
- UDC-UNet (https://github.com/J-FHu/UDCUNet)
- DISCNet (https://github.com/jnjaby/DISCNet)
- Uformer (https://github.com/ZhendongWang6/Uformer)

In addition, UDC-SIT dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). This means that you are allowed to freely utilize, share, and modify this work under the condition of properly attributing the original author, distributing any derived works under the same license, and utilizing it exclusively for non-commercial purposes.
