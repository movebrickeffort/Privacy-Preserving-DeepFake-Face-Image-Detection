import torch.nn as nn
import torch
import torch.quantization
import torch.nn as tnn
import torchvision.transforms as transforms
import random
import datetime
import scipy.io as io
from PIL import Image
import torchvision.datasets as dsets
from tensorboard import summary
from torch.multiprocessing import Manager
import numpy as np
import cv2
import os
from FixedPoint import FXfamily, FXnum
import random


def randomImage(image):
    image_1 = np.random.random(image.shape)
    return image_1

# 读取一张测试
def readImg(image):
    # image = cv2.imread(image)
    # image = cv2.resize(image,(256,256))
    # image = image/255.0
    # image = np.transpose(image, (2, 0, 1))
    # image = np.array(image,np.float32)
    # image = torch.tensor(image, dtype=torch.float32)
    #image = torch.unsqueeze(image, dim=0)

    image_1 = randomImage(image)
    image_2 = image - image_1
    image_1 = torch.tensor(image_1,dtype=torch.float32)
    image_2 = torch.tensor(image_2, dtype=torch.float32)
    # image_1 = torch.unsqueeze(image_1, dim=0)
    # image_2 = torch.unsqueeze(image_2, dim=0)


    return image_1, image_2

if __name__ == '__main__':
    print(-1.2433e+01+1.5746)



