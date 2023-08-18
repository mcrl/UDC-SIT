# UDC-SIT benchmark: SRGAN

## Modifications

+ We apply following normalization to each dataset
  + Feng dataset: apply tonemapping (`x = x / (x + 0.25)`)
  + UDC-SIT dataset: apply normalization (`x = x / 1024`)
+ We do not use upsample blocks because the input size and the output size matches.
  + We attach 0 upsample blocks.

## Dependencies

We use the same environment with ECFNet benchmark. Refer to `README.md` of the experiment.

## Howto

### Data Preparation

+ For Feng dataset, we use `ZTE_new_5` images for input images.
+ Please prepare datasets as following description.

```plain
data
|- Feng
|  |- training
|  |  |- GT
|  |  `- input
|  `- validation # We used the test set of the dataset for validation AND test purpose.
|     |- GT
|     `- input
`- UDC-SIT
   |- training
   |  |- GT
   |  `- input
   |- validation
   |  |- GT
   |  `- input
   `- test
      |- GT
      `- input
```

### Train

```bash
bash train_Feng.sh
bash train_SIT.sh
```

### Test

```bash
bash test_Feng.sh
bash test_SIT.sh
```

## Miscellaneous Notes

+ We perform our training steps with a node with 4 NVIDIA RTX 3090 GPUs.
  + Training phase will consume 8GB of device memory at its peak.
+ We perform our test phase with a single NVIDIA RTX 3090 GPU.
  + TBD: report test memory consumption
+ Our training steps consists of two steps: pretraining generator network, and training GAN.
  + We train each step without stopping.
  + We do not implement options to continue training.

## Acknowledgement

+ Our work is motivated by [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802). We appreciate the authors' effort.
+ Our work is motivated by open-sourced efforts of [SRGAN Pytorch implementation](https://github.com/leftthomas/SRGAN).
We appreciate their effort. For more detail, refer to aforementioned repositories.
