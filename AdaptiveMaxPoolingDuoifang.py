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

def generate_share(Conv, a, b, c):
    # 生成与input形状相同、元素全为1的张量
    A = torch.ones_like(Conv) * a
    B = torch.ones_like(Conv) * b
    C = torch.ones_like(Conv) * c

    # V = np.random.random(Conv.shape)
    V = torch.ones_like(Conv) * random.uniform(0, 1)
    Alpha1 = Conv - A
    Beta1 = V - B

    return A.data, B.data, C.data, V.data, Alpha1.data, Beta1.data

def divideOnEnc(Time, a, b, c):  # compute Pool/(Time1+ Time2)
    V = torch.ones_like(Time) * random.uniform(0, 1)
    A = torch.ones_like(Time) * a
    B = torch.ones_like(Time) * b
    C = torch.ones_like(Time) * c

    # Alpha1 = Pool - A
    # Beta1 = V - B

    Alpha1 = Time - A
    Beta1 = V - B

    return A, B, C, V, Alpha1, Beta1

def maxPoolForServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1):
    #maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    maxpool = nn.AdaptiveMaxPool2d(1)
    (A1, B1, C1, V1, Alpha1, Beta1) = generate_share(input, a1, b1, c1)
    dict_manager.update({'Alpha1': Alpha1, 'Beta1': Beta1})
    # print(Alpha1.shape)
    # print(getsizeof(Alpha1.storage()))
    # print(getsizeof(Beta1.storage()))

    event2.set()
    event1.wait()

    F1 = C1 + B1.mul(Alpha1 + dict_manager['Alpha2']) + A1.mul(Beta1 + dict_manager['Beta2'])
    dict_manager.update({'F1': F1})
    # print(getsizeof(F1.storage()))

    event2.clear()
    event4.set()
    event3.wait()

    F_enc = F1 + dict_manager['F2']

    pool_enc = maxpool(F_enc)
    Times1 = maxpool(V1)

    (A1, B1, C1, V1, Alpha1, Beta1) = divideOnEnc(Times1, a1, b1, c1)
    dict_manager.update({'Alpha1': Alpha1, 'Beta1': Beta1})
    event4.clear()
    event2.set()
    event1.wait()

    F1 = C1 + B1.mul(Alpha1 + dict_manager['Alpha2']) + A1.mul(Beta1 + dict_manager['Beta2'])

    dict_manager.update({'F1': F1})
    event2.clear()
    event4.set()
    event3.wait()

    Times = F1 + dict_manager['F2']
    event4.clear()
    return (pool_enc * V1).div(Times)


def maxPoolForServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2):
    #maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    maxpool = nn.AdaptiveMaxPool2d(1)

    (A2, B2, C2, V2, Alpha2, Beta2) = generate_share(input, a2, b2, c2)
    dict_manager.update({'Alpha2': Alpha2, 'Beta2': Beta2})
    # print(getsizeof(Alpha2.storage()))
    # print(getsizeof(Beta2.storage()))

    event1.set()
    event2.wait()

    F2 = C2 + B2.mul(dict_manager['Alpha1'] + Alpha2) + A2.mul(dict_manager['Beta1'] + Beta2) + \
         (dict_manager['Alpha1'] + Alpha2) * (dict_manager['Beta1'] + Beta2)
    dict_manager.update({'F2': F2})
    # print(getsizeof(F2.storage()))

    event1.clear()
    event3.set()
    event4.wait()

    F_enc = dict_manager['F1'] + F2

    pool_enc = maxpool(F_enc)
    Times2 = maxpool(V2)

    (A2, B2, C2, V2, Alpha2, Beta2) = divideOnEnc(Times2, a2, b2, c2)
    dict_manager.update({'Alpha2': Alpha2, 'Beta2': Beta2})
    event3.clear()
    event1.set()
    event2.wait()

    F2 = C2 + B2.mul(dict_manager['Alpha1'] + Alpha2) + A2.mul(dict_manager['Beta1'] + Beta2) + \
         (dict_manager['Alpha1'] + Alpha2) * (dict_manager['Beta1'] + Beta2)

    dict_manager.update({'F2': F2})
    event1.clear()
    event3.set()
    event4.wait()

    Times = dict_manager['F1'] + F2
    event3.clear()
    return (pool_enc * V2).div(Times)