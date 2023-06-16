# Privacy-Preserving-DeepFake-Face-Image-Detection
## Overview
#### A privacy-preserving Secure DeepFake Detection Network model is designed for the first time. The model uses the additive secret sharing method.

## Prerequisites
#### Ubuntu 18.04
#### NVIDIA GPU+CUDA CuDNN (CPU mode may also work, but untested)
#### Install Torch 1.3.1, torchvision 0.4.2 and dependencies

## Training and Test Details
#### When you train a RGB or YCbCr single-stream model, you should change the input (input.py or input_ycbcr.py) in train.py. The corresponding part should also be modified during testing. When testing the dual-stream model, the RGB image and its YCbCr image should be input together. When running the code under the ciphertext, run main.py
