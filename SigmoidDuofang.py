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
import DivDuofang
import ExpDuofang
import randomC
import math
from multiprocessing import Process, Queue, Pipe
import GenerateShare
import main
import xgg_model
import input

def serverSigmoid1(u1,c1_mul,c1_res,t1,a1,b1,c1,dict_manager,event1,event2):
    a1,b1,c1,t1,c1_res,c1_mul = GenerateShare.generate_share2(u1,a1,b1,c1,t1,c1_res,c1_mul)
    #print("指数开始")
    event1.clear()
    event2.clear()
    e_u1 = ExpDuofang.expServer1(-u1,c1_mul,c1_res,dict_manager,event1,event2)
    #print("指数完成")
    event1.clear()
    event2.clear()
    x1 = torch.ones_like(u1) * 1/2
    y1 = e_u1 + torch.ones_like(u1) * 1/2
    res1 = DivDuofang.serverDiv1(x1,y1,t1,a1,b1,c1,dict_manager,event1,event2)

    print(type(res1))
    #print('sigmoid1完成')

    #print("res1_sigmoid: ",res1)
    #dict_manager.update({"res1":res1})
    return res1


def serverSigmoid2(u2,c2_mul,c2_res,t2,a2,b2,c2,dict_manager,event1,event2):
    a2, b2, c2, t2, c2_res, c2_mul = GenerateShare.generate_share2(u2, a2, b2, c2, t2, c2_res, c2_mul)
    event1.clear()
    event2.clear()
    #print('sigmoid2开始')
    e_u2 = ExpDuofang.expServer2(-u2, c2_mul, c2_res, dict_manager,event1,event2)
    #print('sigmoid2 exp结束')
    event1.clear()
    event2.clear()
    x2 = torch.ones_like(u2) * 1/2
    y2 = e_u2 + torch.ones_like(u2) * 1/2
    #print('sigmoid2 div开始')
    res2 = DivDuofang.serverDiv2(x2, y2, t2, a2, b2, c2, dict_manager,event1,event2)
    #print('sigmoid2 div结束')

    #print(type(res2))
   # print('sigmoid2完成')
    return res2

    #print("res2_sigmoid: ", res2)
    #dict_manager.update({"res2": res2})

def serverSigmoid(u1,u2):

    c1_mul = randomC.random_c()
    c2_mul = randomC.random_c()

    c = c1_mul * c2_mul
    c1_res = randomC.random_c()
    c2_res = c - c1_res

    # c1_mul = torch.ones_like(u1) * c1_mul
    # c2_mul = torch.ones_like(u1) * c2_mul
    # c1_res = torch.ones_like(u1) * c1_res
    # c2_res = torch.ones_like(u1) * c2_res

    t = randomC.random_c()
    t1 = randomC.random_c()
    t2 = t - t1

    # t1 = torch.ones_like(u1) * t1
    # t2 = torch.ones_like(u1) * t2


    a = randomC.random_c()
    a1 = randomC.random_c()
    a2 = a - a1

    b = randomC.random_c()
    b1 = randomC.random_c()
    b2 = b - b1

    c = a * b
    c1 = randomC.random_c()
    c2 = c - c1

    # a1 = torch.ones_like(u1) * a1
    # b1 = torch.ones_like(u1) * b1
    # c1 = torch.ones_like(u1) * c1
    # a2 = torch.ones_like(u1) * a2
    # b2 = torch.ones_like(u1) * b2
    # c2 = torch.ones_like(u1) * c2

    event1 = torch.multiprocessing.Event()
    event2 = torch.multiprocessing.Event()
    dict_manager = Manager().dict()

    #print("kaishi")

    process1 = Process(target=serverSigmoid1,args=(u1,c1_mul,c1_res,t1,a1,b1,c1,dict_manager,event1,event2))
    process2 = Process(target=serverSigmoid2,args=(u2,c2_mul,c2_res,t2,a2,b2,c2,dict_manager,event1,event2))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    res1 = dict_manager['res1']
    res2 = dict_manager['res2']

    return res1+res2






if __name__ == '__main__':
    #
    # x = np.array([1,2,3])
    # x = torch.tensor(x,dtype=torch.float32)
    # y = np.array([2,3,4])
    # y = torch.tensor(y,dtype=torch.float32)

    xgg = xgg_model.xception_agg()
    xgg.load_state_dict(torch.load('40cnn.pkl', map_location='cpu'))
    # 将这个 new_nodel 传给两个进程
    # 用管道来传递数据
    new_model = nn.Sequential(*list(xgg.children()))[:]
    net = new_model[0]
    Bn1 = new_model[1]
    Bn1.eval()
    image_1,image_2 = input.readImg('test.png')
    # print(image_1.shape)
    # x1 = net(image_1)
    # x2 = net(image_2)
    # print(x1+x2)

    image = cv2.imread('test.png')
    image = cv2.resize(image, (256, 256))
    image = image/255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.array(image, np.float32)
    # image = torch.unsqueeze(image, dim=0)
    image = torch.tensor(image,dtype=torch.float32)

    #print(image_2)
    image = torch.unsqueeze(image, dim=0)
    x1,x2 = main.xce_agg(image_1, image_2, new_model)
    x1 = torch.tensor(x1,dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)

    res = serverSigmoid(x1,x2)
    print("res:",res)


    #
    # x3 = net(image)
    # bn = Bn1(x3)
    # relu = new_model[2]
    # Conv2 = new_model[3]
    # r = relu(bn)
    # F2 = Conv2(r)
    # BN2 = new_model[4]
    # BN2.eval()
    # bn2 = BN2(F2)
    # block1 = nn.Sequential(*list(new_model[5].children()))[:]
    # block1_1 = block1[0](bn2)
    # block1_2 = block1[1].eval()
    # block1_2 = block1_2(block1_1)
    # block1_3 = block1[2](block1_2)
    # ChannelAttention = nn.Sequential(*list(block1[3].children()))[:]
    # avgPool = ChannelAttention[0](block1_3)
    # maxPool = ChannelAttention[1](block1_3)
    # fc1_avg = ChannelAttention[2](avgPool)
    # relu1_avg = ChannelAttention[3](fc1_avg)
    # fc2_avg = ChannelAttention[4](relu1_avg)
    #
    # fc1_max = ChannelAttention[2](maxPool)
    # relu1_max = ChannelAttention[3](fc1_max)
    # fc2_max = ChannelAttention[4](relu1_max)
    #
    # x = fc2_avg + fc2_max
    # #sigmoid1 = ChannelAttention[5](x)
    #
    # print("最后结果 ",x.shape)
    # print(x)

    #print(ProbabilityValue)

    # serverSigmoid(x,y)
    #
    # s = torch.nn.Sigmoid()
    #
    # #print("正确sigmoid:", math.exp(2) / (math.exp(2) - 1))
    # print("正确sigmoid:", s(x+y))
    #
    # print("正确sigmoid2:",1/(torch.exp(-x-y) + 1))

