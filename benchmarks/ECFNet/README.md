# UDC-SIT benchmark: ECFNet

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
python test.py
```

## Miscellaneous Notes

+ Our training phase was performed with a node with 4 NVIDIA RTX 3090 GPUs.
  + Training phase will consume 22GB of device memory at its peak.
+ Our test phase was performed with a single NVIDIA RTX 3090 GPU.
  + Test phase will consume 20GB of device memory at its peak.
