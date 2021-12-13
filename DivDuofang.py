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
import SecMulDuofang
# 输入[x]1 [x]2 [y]1 [y]2
# 输出【x/y】1 [x/y]2

def serverDiv1(x1_res,y1_res,t1,a1,b1,c1,dict_manager,event1,event2):
    #print("server1")
    #print(type(t1))
    ty1 = SecMulDuofang.SecMulServer1(y1_res, t1, a1, b1, c1, dict_manager,event1,event2)
    event2.clear()
    event1.clear()
    tx1 = SecMulDuofang.SecMulServer1(x1_res,t1,a1,b1,c1,dict_manager,event1,event2)

    #print("ty1:",ty1)
    event2.clear()
    event1.clear()

    #print("ty1:",ty1)
    #p1.send([ty1])
    dict_manager.update({'ty1':ty1.detach()})

    event2.set()
    event1.wait()
    #ty2 = p1.recv()[0]
    ty2 = dict_manager['ty2']

    ty = ty1+ty2
    #print("ty:",ty)
    res1 = tx1/ty
    #print("divType", type(res1))
    #print("res1:",res1)
    #print("res1",res1)
    #print("server1_div 完成")
    #print("111:",res1.detach().numpy())
    #p1.send(res1.detach().numpy())
    return res1

def serverDiv2(x2_res,y2_res,t2,a2,b2,c2,dict_manager,event1,event2):
    #print("div server2")
    #print(type(t2))
    ty2 = SecMulDuofang.SecMulServer2(y2_res, t2, a2, b2, c2, dict_manager,event1,event2)
    #print("div server2 1")
    event2.clear()
    event1.clear()
    tx2 = SecMulDuofang.SecMulServer2(x2_res,t2,a2,b2,c2,dict_manager,event1,event2)
    #print("div server2 2")


    #print("ty2:",ty2)
    event2.clear()
    event1.clear()
    #print("ty2:",ty2)
    #p2.send([ty2])
    dict_manager.update({'ty2':ty2.detach()})
    event1.set()
    event2.wait()
    #ty1 = p2.recv()[0]
    ty1 = dict_manager['ty1']
    ty = ty1+ty2

    res2 = tx2/ty
    #print("res2:",res2)
    #print("res2:",res2)
    #print("server2_div 完成")
    #print("divType",type(res2))
    #p2.send(res2.detach().numpy())
    return res2