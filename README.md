# [CORE-Net: A cross-modal orthogonal representation enhancement network for low-altitude multispectral object detection](https://doi.org/10.1371/journal.pone.0340499)

Official PyTorch implementation of CORE-Net.

## Installation

```
pip install -e .
```

## Training

```
python train.py
```

## Acknowledgement

The code base is built with [Ultralytics](https://github.com/ultralytics/ultralytics).

Gratitude is extended to all contributors who supported this study through their valuable assistance and insights.

**Note:** Dual-modal input training requires significantly more memory. If the training is automatically terminated with a "Killed" prompt, it may indicate a memory overflow. Monitor memory usage during training and adjust hyperparameters based on the device’s specifications.
