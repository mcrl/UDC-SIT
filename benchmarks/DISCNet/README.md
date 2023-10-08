# UDC-SIT benchmark: DISCNet

## Modifications

+ Dataset

  + We apply normalization with `1023`, while the original work apply tone mapping(`x / (x + 0.25)`).

+ Model

  + We add one channel to the input channel of the first block and the output channel of the last block.

+ Validation Loop

  + We change image saving function to support 4-channeled `.dng` format.

+ Train Option

  + We omit perceptual loss, because the image channel count does not match with the loss model's input channel.

## Dependencies

+ Python = 3.8
+ PyTorch = 2.0.1
+ mmcv = 1.7.1
+ CUDA = 11.7

## Installation

+ Install Python, CUDA, and PyTorch
+ Install `MMCV==1.7.1` with [our repository](https://github.com/mcrl/mmcv-for-UDC-SIT)

```bash
git clone git@github.com:mcrl/mmcv-for-UDC-SIT.git
cd mmcv-for-UDC-SIT.git
pip install -r requirements/optional.txt
pip install -e . -v
```

> Installation with `pip`, `conda`, `mim` did not work since Aug. 2023. We ship our workaround.

+ Run following command

```bash
cd UDC-SIT/benchmarks/DISCNet
pip install -r requirements.txt
bash make.sh
```

## Howto

### Data Preparation

+ Identify where the dataset resides in your filesystem.
+ Run to generate meta-info-files

```bash
python generate-metainfo.py \
  --train-gt /path/to/train/gt \
  --val-gt /path/to/val/gt \
  --test-gt /path/to/test/gt
```

You may modify `psf.npy` file location. Refer to `python generate-metainfo.py --help`.

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

## License and Acknowledgement

This work is licensed under MIT License. The codes are mainly from following repositories.
For more information, refer to [original work](https://github.com/jnjaby/DISCNet)
and [BasicSR](https://github.com/XPixelGroup/BasicSR).
