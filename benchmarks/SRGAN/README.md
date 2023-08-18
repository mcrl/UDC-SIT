# UDC-SIT benchmark: SRGAN

## Modifications

+ TBD

## Dependencies

We use the same environment with ECFNet benchmark. Refer to `README.md` of the experiment.

## Howto

### Data Preparation

TBD. The descrtiption following is not updated!

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

+ We perform our training steps with a node with 4 NVIDIA RTX 3090 GPUs.
  + Training phase will consume 8GB of device memory at its peak.
+ We perform our test phase with a single NVIDIA RTX 3090 GPU.
  + TBD: report test memory consumption
+ Our training steps consists of two steps: pretraining generator network, and training GAN.
  + We train each step without stopping.
  + We do not implement options to continue training.

## Acknowledgement

+ TBD
