# UDC-SIT benchmark: UDC-UNet

## Modifications

+ Dataset

  + We apply normalization with `1023`, while the original work apply tone mapping(`x / (x + 0.25)`).
  + We modify [util funtion](basicsr/data/util.py) `paired_paths_from_folder` to accept `.npy` files only.

+ Model

  + We add one channel to the input channel of the first block and the output channel of the last block.

+ Loss Function

  + Instead of the original work's `map_L1Loss`, we apply `clamp_L1Loss`, because our dataset is LDR images.

+ Validation Loop

  + We change image saving function to support 4-channeled `.dng` format.
  + We do not use `lpips` for metric, becasue the image channel count does not match with the metric model's input channel.

+ Train Option

  + We omit perceptual loss, because the image channel count does not match with the loss model's input channel.

## Dependencies

+ Python = 3.8
+ PyTorch = 2.0.1
+ mmcv = 1.7.1
+ CUDA = 11.7

## Installation

+ Install Python, CUDA, and PyTorch
+ Run following command

```bash
pip install -r requirements.txt
python setup.py develop
```

## Howto

### Data Preparation

+ Identify where the dataset resides in your filesystem.

### Train

+ Modify `option/train/train.yml` to specify data path
+ Run command

```bash
bash train.sh
```

### Test

+ Modify `option/test/test.yml` to specify data path
+ Run command

```bash
bash test.sh
```

## Acknowledgement and License

This work is licensed under MIT License. The code is mainly modified from [Original Work](https://github.com/J-FHu/UDCUNet)
and [BasicSR](https://github.com/XPixelGroup/BasicSR). Refer to original repositories for more detail.
