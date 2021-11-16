# Privacy-Preserving-DeepFake-Face-Image-Detection
## Overview
#### In order to detect more and more DeepFake faces on the Internet, some models have been successfully proposed. However, these models focus on plaintext DeepFake faces, ignoring the problem of privacy preservation. Thus, in this paper, we design a privacy-preserving Secure DeepFake Detection Network (SecDFDNet) model for the first time. The model uses the additive secret sharing method to detect DeepFake faces. Specifically, firstly, some multiparty secure protocols are designed for non-linear activate functions, i.e., secure sigmoid protocol SecSigm, secure channel attention protocol SecChannel, and secure spatial attention protocol SecSpatial. Their securities are proofed in theoretical. Then, a privacy-preserving model SecDFDNet is proposed for secure DeepFake face detection by using the trained plaintext DeepFake detection network (DFDNet) and designed secure protocols. The experimental results show that the proposed SecDFDNet  can securely detect DeepFake faces with the similar accuracy as the plaintext DFDNet.

## Prerequisites
#### Ubuntu 18.04
#### NVIDIA GPU+CUDA CuDNN (CPU mode may also work, but untested)
#### Install Torch 1.3.1, torchvision 0.4.2 and dependencies

## Training and Test Details
#### When you train a RGB or YCbCr single-stream model, you should change the input (input.py or input_ycbcr.py) in train.py. The corresponding part should also be modified during testing. When testing the dual-stream model, the RGB image and its YCbCr image should be input together. When running the code under the ciphertext, run main.py

## Related Works
#### [1] Chen B J, Liu X, and Zheng Y H. A robust GAN-generated face detection method based on dual-color spaces and an improved Xception. IEEE Transactions on Circuits and Systems for Video Technology, 2021. DOI: 10.1109/TCSVT.2021.3116679.
