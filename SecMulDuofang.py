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
import GenerateShare
import randomC
from multiprocessing import Process, Queue, Pipe

## 输入 [x]i [y]i 输出[xy]i

def SecMulServer1(x1,y1,a1,b1,c1,dict_manager,event1,event2):
    event2.clear()
    event1.clear()
    #a1, b1, c1, _, Alpha1, Beta1 = GenerateShare.generate_share(x1, a1, b1, c1)
    e1 = x1 - a1
    f1 = y1 - b1
    #print("secmul1")
    #p1.send([e1,f1])
    dict_manager.update({'e1':e1.detach(),'f1':f1.detach()})
    event2.set()
    event1.wait()
    #print("secmul2")
    #e_f = p1.recv()
    #print("secmul3")

    #e2 = e_f[0]
    #f2 = e_f[1]
    e2 = dict_manager['e2']
    f2 = dict_manager['f2']
    event1.clear()
    #print(type(e2))

    e = e1 + e2
    f = f1 + f2

    res1 = c1+b1*e+a1*f+e*f
    #p1.send(res1.detach().numpy())
    #print(res1)

    return res1


def SecMulServer2(x2, y2, a2, b2, c2, dict_manager,event1,event2):
    event2.clear()
    event1.clear()
    #a2, b3, c3, _, Alpha1, Beta1 = GenerateShare.generate_share(x2, a2, b2, c2)
    e2 = x2 - a2
    f2 = y2 - b2

    #p2.send([e2, f2])
    dict_manager.update({'e2':e2.detach(),'f2':f2.detach()})
    event1.set()
    event2.wait()
    # e_f = p2.recv()
    #
    # e1 = e_f[0]
    #
    # f1 = e_f[1]
    e1 = dict_manager['e1']
    f1 = dict_manager['f1']

    event2.clear()

    e = e1 + e2
    f = f1 + f2

    res2 = c2 + b2 * e + a2 * f
    #print("res2: ",res2)
    #p2.send(res2.detach().numpy())
    #print(res2)
    return res2

if __name__ == '__main__':
    a = randomC.random_c()
    a1 = randomC.random_c()
    a2 = a - a1

    b = randomC.random_c()
    b1 = randomC.random_c()
    b2 = b - b1

    c = a * b
    c1 = randomC.random_c()
    c2 = c - c1

    event1 = torch.multiprocessing.Event()
    event2 = torch.multiprocessing.Event()
    dict_manager = Manager().dict()
    event1.clear()
    event2.clear()


    process1 = Process(target=SecMulServer1,args=(5.4140013e-8,-18.896248,a1,b1,c1,dict_manager,event1,event2))
    process2 = Process(target=SecMulServer2,args=(-6.80337608e-9, 2.0302591, a2, b2, c2, dict_manager, event1, event2))

    process1.start()
    process2.start()

    process1.join()
    process2.join()
    print(5.4140013e-8+-6.80337608e-9)
    print(-18.896248 +2.0302591)

    print(-16.865978*4.7337e-8)
    print(0.22429526081733897+-0.22429605919653195)

