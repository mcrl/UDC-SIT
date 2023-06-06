# UDC-SIT benchmark: ECFNet

## Modifications

+ We apply normalization, instead of tone mapping. Check [`_tonemap`](datasets/dataset_pairs_npy.py) function.
+ We add one channel to the input channel of the first block and the output channel of the last block.
+ We dynamically change `base_channel` of the model with `in_nc * 8`. As UDC-SIT images have 4 channels, `base_channel` is set to `32`.
+ We provide our own version of train script, as we wrote the train script from the scratch.
+ The modification regarding batchsizes are descripted in [`train-ecfnet.sh`](train-ecfnet.sh).
+ We specify several modules that will not save intermediate results in the device. Check this modules [here](train.py).

## Dependencies

+ Python>=3.8
+ PyTorch>=2.0.1
+ CUDA>=11.7

> Other required packages are listed in `requirements.txt`

## Howto

### Data Preparation

Place UDC-SIT dataset into `data` directory as below:

```plain
data
|- training
|  |- input
|  `- GT
|- validation
|  |- input
|  `- GT
`- test
   |- input
   `- GT
```

### Train

```bash
bash train-ecfnet.sh
```

### Test

```bash
bash test.sh
```

## Miscellaneous Notes

+ Our training phase was performed with a node with 4 NVIDIA RTX 3090 GPUs.
  + Training phase will consume 22GB of device memory at its peak.
+ Our test phase was performed with a single NVIDIA RTX 3090 GPU.
  + Test phase will consume 20GB of device memory at its peak.

## Acknowledgement

+ Our work is motivated by open-sourced efforts of [MIMO-UNet](https://github.com/chosj95/MIMO-UNet)
and [ECFNet](https://github.com/zhuyr97/ECFNet). We appreciate their effort.
For more detail, refer to aforementioned repositories.
