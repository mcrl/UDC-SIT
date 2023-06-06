# UDC-SIT benchmark: Uformer

## Modifications

+ Dataset

  + We apply normalization with `1023`, while the original work apply normalization with `255`.

+ Model

  + We add one channel to the input channel of the first block and the output channel of the last block.

+ Validation Loop

  + We change image saving function to support 4-channeled `.dng` format.


## Installation

+ Install Python, CUDA, and PyTorch
+ Run following command

```bash
pip install -r requirements.txt
```


### Train

+ Modify `script/train_udcsit.sh` to specify data path
+ Run command

```bash
sh ./script/train_udcsit.sh
```

### Test

+ Modify `script/test_udcsit.sh` to specify data path
+ Run command

```bash
sh ./script/test_udcsit.sh
```

## License and Acknowledgement

This work is licensed under MIT License. The codes are mainly from following repositories.
For more information, refer to [original work](https://github.com/ZhendongWang6/Uformer), [MIRNet](https://github.com/swz30/MIRNet)
and [SwinTransformer](https://github.com/microsoft/Swin-Transformer).
