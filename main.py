import xgg_model
import torch.nn as nn
import torch
import torch.quantization
import torch.nn as tnn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as dsets
from torch.multiprocessing import Manager
import numpy as np
import input
from torch.multiprocessing import Process, Queue, Pipe
import randomC
import ReluDuofang
import AdaptiveMaxPoolingDuoifang
import SigmoidDuofang
import SecMulDuofang
import TorchMax
import MaxPoolingDuofang
import torch.nn.functional as F
from key import key

def GetBN(beforeBn,channel):
    newBN = nn.BatchNorm2d(channel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    newBN.eval()

    newBNGrammer = beforeBn.weight
    newBNBeita = beforeBn.bias / 2.0
    newBNMean = beforeBn.running_mean / 2.0
    newBNVar = beforeBn.running_var

    newBN.weight = nn.Parameter(newBNGrammer)
    newBN.bias = nn.Parameter(newBNBeita)
    newBN.running_mean = newBNMean
    newBN.running_var = newBNVar

    return newBN

def ChannelAttentionServer1(x,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul):
    avgPool = ChannelAttention[0](x)
    maxPool = AdaptiveMaxPoolingDuoifang.maxPoolForServer1(x, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    fc1_avg = ChannelAttention[2](avgPool)
    relu_avg = ReluDuofang.reluForServer1(fc1_avg,dict_manager, event1, event2, event3, event4, a1, b1, c1)
    fc2_avg = ChannelAttention[4](relu_avg)

    fc1_max = ChannelAttention[2](maxPool)
    relu_max = ReluDuofang.reluForServer1(fc1_max,dict_manager, event1, event2, event3, event4, a1, b1, c1)
    fc2_max = ChannelAttention[4](relu_max)
    fc2 = fc2_avg + fc2_max

    # 对 fc2 Sigmoid
    sigmoid1 = SigmoidDuofang.serverSigmoid1(fc2,c1_mul,c1_res,t1,a1,b1,c1,dict_manager1,event1,event2)
    A = torch.ones_like(sigmoid1) * a1
    B = torch.ones_like(sigmoid1) * b1
    C = torch.ones_like(sigmoid1) * c1
    event1.clear()
    event2.clear()
    res = SecMulDuofang.SecMulServer1(sigmoid1,x,A,B,C,dict_manager,event1,event2)
    return res
def ChannelAttentionServer2(x,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul):
    avgPool = ChannelAttention[0](x)
    maxPool = AdaptiveMaxPoolingDuoifang.maxPoolForServer2(x, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    fc1_avg = ChannelAttention[2](avgPool)
    relu_avg = ReluDuofang.reluForServer2(fc1_avg, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    fc2_avg = ChannelAttention[4](relu_avg)

    fc1_max = ChannelAttention[2](maxPool)
    relu_max = ReluDuofang.reluForServer2(fc1_max, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    fc2_max = ChannelAttention[4](relu_max)

    fc2 = fc2_avg + fc2_max

    # 对 fc2 Sigmoid
    sigmoid1 = SigmoidDuofang.serverSigmoid2(fc2,c2_mul,c2_res,t2,a2,b2,c2,dict_manager1,event1,event2)
    A = torch.ones_like(sigmoid1) * a2
    B = torch.ones_like(sigmoid1) * b2
    C = torch.ones_like(sigmoid1) * c2
    event1.clear()
    event2.clear()
    res = SecMulDuofang.SecMulServer2(sigmoid1,x,A,B,C,dict_manager,event1,event2)

    return res

def SpatialAttentionServer1(x,Spatialattention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul):
    #print("x Shape:", x.shape)
    avg_out = torch.mean(x,dim=1,keepdim=True)
    max_out = TorchMax.maxPoolForServer1(x,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    x_cat = torch.cat([avg_out,max_out],dim = 1)
    sp = Spatialattention[0](x_cat)
    sigmoid1 = SigmoidDuofang.serverSigmoid1(sp,c1_mul,c1_res,t1,a1,b1,c1,dict_manager1,event1,event2)
    A = torch.ones_like(sigmoid1) * a1
    B = torch.ones_like(sigmoid1) * b1
    C = torch.ones_like(sigmoid1) * c1

    res = SecMulDuofang.SecMulServer1(sigmoid1,x,A,B,C,dict_manager,event1,event2)

    return res


def SpatialAttentionServer2(x, Spatialattention, dict_manager, dict_manager1, event1, event2, event3, event4, a2, b2,
                            c2, t2, c2_res, c2_mul):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out = TorchMax.maxPoolForServer2(x, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    x_cat = torch.cat([avg_out, max_out], dim=1)
    sp = Spatialattention[0](x_cat)
    sigmoid1 = SigmoidDuofang.serverSigmoid2(sp, c2_mul, c2_res, t2, a2, b2, c2, dict_manager1, event1, event2)
    A = torch.ones_like(sigmoid1) * a2
    B = torch.ones_like(sigmoid1) * b2
    C = torch.ones_like(sigmoid1) * c2

    res = SecMulDuofang.SecMulServer2(sigmoid1, x, A, B, C, dict_manager,event1, event2)

    return res


def Block0Server2(x2, Convd1, BN1, Convd2, BN2, a2, b2, c2, event1, event2, event3, event4, dict_manager):
    Feature1 = Convd1(x2)
    bn1 = GetBN(BN1, 32)(Feature1)
    relu1 = ReluDuofang.reluForServer2(bn1, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    Feature2 = Convd2(relu1)
    bn2 = GetBN(BN2, 64)(Feature2)
    relu2 = ReluDuofang.reluForServer2(bn2, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    return relu2


def Block0Server1(x1,Convd1,BN1,Convd2,BN2,a1,b1,c1,event1,event2,event3,event4,dict_manager):
    Feature1 = Convd1(x1)
    bn1 = GetBN(BN1,32)(Feature1)
    relu1 = ReluDuofang.reluForServer1(bn1, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    Feature2 = Convd2(relu1)
    bn2 = GetBN(BN2, 64)(Feature2)
    relu2 = ReluDuofang.reluForServer1(bn2, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    return relu2
def Block1Server1(x1,block,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul):

    Conv2d1 = block[0](x1)
    bn1 = GetBN(block[1],128)(Conv2d1)
    #relu1 = ReluDuofang.reluForServer1(bn1,dict_manager, event1, event2, event3, event4, a1, b1, c1)
    rep = nn.Sequential(*list(block[5].children()))[:]
    rep1 = rep[0](x1)
    rep2 = GetBN(rep[1],128)(rep1)
    rep3 = ReluDuofang.reluForServer1(rep2,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep4 = rep[3](rep3)
    rep5 = GetBN(rep[4],128)(rep4)
    rep6 = MaxPoolingDuofang.maxPoolForServer1(rep5,dict_manager,event1,event2,event3,event4,a1,b1,c1)

    # 注意力机制
    ChannelAttention = nn.Sequential(*list(block[3].children()))[:]
    # 走个注意力机制的函数
    channelattention = ChannelAttentionServer1(rep6,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    SpatialAttention = nn.Sequential(*list(block[4].children()))[:]
    spatialattention = SpatialAttentionServer1(channelattention,SpatialAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    res = spatialattention + bn1
    #print(res.shape)
    #print("server1: ",res)

    return res

def Block1Server2(x2,block,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul):
    Conv2d1 = block[0](x2)
    bn1 = GetBN(block[1],128)(Conv2d1)
    #relu1 = ReluDuofang.reluForServer2(bn1,dict_manager, event1, event2, event3, event4, a2, b2, c2)
    rep = nn.Sequential(*list(block[5].children()))[:]
    rep1 = rep[0](x2)
    rep2 = GetBN(rep[1],128)(rep1)
    rep3 = ReluDuofang.reluForServer2(rep2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep4 = rep[3](rep3)
    rep5 = GetBN(rep[4],128)(rep4)
    rep6 = MaxPoolingDuofang.maxPoolForServer2(rep5,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    ChannelAttention = nn.Sequential(*list(block[3].children()))[:]
    # 走个注意力机制的函数
    channelattention = ChannelAttentionServer2(rep6,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul)
    SpatialAttention = nn.Sequential(*list(block[4].children()))[:]
    spatialattention = SpatialAttentionServer2(channelattention, SpatialAttention, dict_manager, dict_manager1, event1,
                                               event2, event3, event4, a2, b2, c2, t2, c2_res, c2_mul)
    res = spatialattention+bn1
    #print("server2: ",res)

    return res

def Block2Server1(x1,block,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul,bnChannel):
    Conv2d1 = block[0](x1)
    bn1 = GetBN(block[1],bnChannel)(Conv2d1)
    #relu1 = ReluDuofang.reluForServer1(bn1,dict_manager, event1, event2, event3, event4, a1, b1, c1)
    rep = nn.Sequential(*list(block[5].children()))[:]

    rep0 = ReluDuofang.reluForServer1(x1,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep1 = rep[1](rep0)
    rep2 = GetBN(rep[2],bnChannel)(rep1)
    rep3 = ReluDuofang.reluForServer1(rep2,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep4 = rep[4](rep3)
    rep5 = GetBN(rep[5],bnChannel)(rep4)
    rep6 = MaxPoolingDuofang.maxPoolForServer1(rep5,dict_manager,event1,event2,event3,event4,a1,b1,c1)

    # 注意力机制
    ChannelAttention = nn.Sequential(*list(block[3].children()))[:]
    # 走个注意力机制的函数
    channelattention = ChannelAttentionServer1(rep6,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    SpatialAttention = nn.Sequential(*list(block[4].children()))[:]
    spatialattention = SpatialAttentionServer1(channelattention,SpatialAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    res = spatialattention + bn1
    # print(res.shape)
    # print("server1: ",res)

    return res

def Block2Server2(x2,block,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul,bnChannel):
    Conv2d1 = block[0](x2)
    bn1 = GetBN(block[1],bnChannel)(Conv2d1)
    #relu1 = ReluDuofang.reluForServer2(bn1,dict_manager, event1, event2, event3, event4, a2, b2, c2)
    rep = nn.Sequential(*list(block[5].children()))[:]
    rep0 = ReluDuofang.reluForServer2(x2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep1 = rep[1](rep0)
    rep2 = GetBN(rep[2],bnChannel)(rep1)
    event1.clear()
    event2.clear()
    event3.clear()
    event4.clear()
    rep3 = ReluDuofang.reluForServer2(rep2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep4 = rep[4](rep3)
    rep5 = GetBN(rep[5],bnChannel)(rep4)
    rep6 = MaxPoolingDuofang.maxPoolForServer2(rep5,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    ChannelAttention = nn.Sequential(*list(block[3].children()))[:]
    # 走个注意力机制的函数
    channelattention = ChannelAttentionServer2(rep6,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul)
    SpatialAttention = nn.Sequential(*list(block[4].children()))[:]
    spatialattention = SpatialAttentionServer2(channelattention, SpatialAttention, dict_manager, dict_manager1, event1,
                                               event2, event3, event4, a2, b2, c2, t2, c2_res, c2_mul)
    res = spatialattention+bn1
    # print("server2: ",res)

    return res

# 不用maxpooling降维的
def Block3Server1(x1,block,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul,bnChannel):

    #relu1 = ReluDuofang.reluForServer1(bn1,dict_manager, event1, event2, event3, event4, a1, b1, c1)
    rep = nn.Sequential(*list(block[3].children()))[:]
    rep0 = ReluDuofang.reluForServer1(x1,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep1 = rep[1](rep0)
    rep2 = GetBN(rep[2],bnChannel)(rep1)
    rep3 = ReluDuofang.reluForServer1(rep2,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep4 = rep[4](rep3)
    rep5 = GetBN(rep[5],bnChannel)(rep4)
    rep6 = ReluDuofang.reluForServer1(rep5,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep7 = rep[6](rep6)
    rep8 = GetBN(rep[7], bnChannel)(rep7)

    # 注意力机制
    ChannelAttention = nn.Sequential(*list(block[1].children()))[:]
    # 走个注意力机制的函数
    channelattention = ChannelAttentionServer1(rep8,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    SpatialAttention = nn.Sequential(*list(block[2].children()))[:]
    spatialattention = SpatialAttentionServer1(channelattention,SpatialAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    res = spatialattention + x1
    #print(res.shape)
    #print("server1: ",res)
    # print("server1:",res)

    return res

def Block3Server2(x2,block,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul,bnChannel):

    #relu1 = ReluDuofang.reluForServer2(bn1,dict_manager, event1, event2, event3, event4, a2, b2, c2)
    rep = nn.Sequential(*list(block[3].children()))[:]
    #print(rep)
    rep0 = ReluDuofang.reluForServer2(x2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep1 = rep[1](rep0)
    rep2 = GetBN(rep[2],bnChannel)(rep1)
    rep3 = ReluDuofang.reluForServer2(rep2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep4 = rep[4](rep3)
    rep5 = GetBN(rep[5],bnChannel)(rep4)
    rep6 = ReluDuofang.reluForServer2(rep5, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    rep7 = rep[6](rep6)
    rep8 = GetBN(rep[7], bnChannel)(rep7)
    ChannelAttention = nn.Sequential(*list(block[1].children()))[:]
    # 注意力机制的函数
    channelattention = ChannelAttentionServer2(rep8,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul)
    SpatialAttention = nn.Sequential(*list(block[2].children()))[:]
    spatialattention = SpatialAttentionServer2(channelattention, SpatialAttention, dict_manager, dict_manager1, event1,
                                               event2, event3, event4, a2, b2, c2, t2, c2_res, c2_mul)
    res = spatialattention+x2


    return res

# 原始的第16个block
def Block4Server1(x1,block,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul,bnChannel1,bnChannel2):
    print("block4 开始")
    Conv2d1 = block[0](x1)
    bn1 = GetBN(block[1],bnChannel2)(Conv2d1)

    rep = nn.Sequential(*list(block[5].children()))[:]
    rep0 = ReluDuofang.reluForServer1(x1,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep1 = rep[1](rep0)
    rep2 = GetBN(rep[2],bnChannel1)(rep1)
    rep3 = ReluDuofang.reluForServer1(rep2,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    rep4 = rep[4](rep3)
    rep5 = GetBN(rep[5],bnChannel2)(rep4)
    rep6 = MaxPoolingDuofang.maxPoolForServer1(rep5,dict_manager,event1,event2,event3,event4,a1,b1,c1)

    # 注意力机制
    ChannelAttention = nn.Sequential(*list(block[3].children()))[:]
    # 注意力机制的函数
    channelattention = ChannelAttentionServer1(rep6,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    SpatialAttention = nn.Sequential(*list(block[4].children()))[:]
    spatialattention = SpatialAttentionServer1(channelattention,SpatialAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    res = spatialattention + bn1
    # print("res.shape")
    # print("server1: ",bn1)

    return res

def Block4Server2(x2,block,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul,bnChannel1,bnChannel2):
    Conv2d1 = block[0](x2)
    bn1 = GetBN(block[1],bnChannel2)(Conv2d1)
    #relu1 = ReluDuofang.reluForServer2(bn1,dict_manager, event1, event2, event3, event4, a2, b2, c2)
    rep = nn.Sequential(*list(block[5].children()))[:]
    rep0 = ReluDuofang.reluForServer2(x2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep1 = rep[1](rep0)
    rep2 = GetBN(rep[2],bnChannel1)(rep1)
    rep3 = ReluDuofang.reluForServer2(rep2,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    rep4 = rep[4](rep3)
    rep5 = GetBN(rep[5],bnChannel2)(rep4)
    rep6 = MaxPoolingDuofang.maxPoolForServer2(rep5,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    ChannelAttention = nn.Sequential(*list(block[3].children()))[:]
    # 走个注意力机制的函数
    channelattention = ChannelAttentionServer2(rep6,ChannelAttention,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul)
    SpatialAttention = nn.Sequential(*list(block[4].children()))[:]
    spatialattention = SpatialAttentionServer2(channelattention, SpatialAttention, dict_manager, dict_manager1, event1,
                                               event2, event3, event4, a2, b2, c2, t2, c2_res, c2_mul)
    res = spatialattention+bn1
    #print("server2: ",bn1)

    return res


def Server2(new_model,x2,p1,p2,p3,p4,a2,b2,c2,t2,c2_res,c2_mul,event1,event2,event3,event4,dict_manager,dict_manager1):
    Convd1 = new_model[0]
    BN1 = new_model[1]
    Convd2 = new_model[3]
    BN2 = new_model[4]
    res_block0 = Block0Server2(x2, Convd1, BN1, Convd2, BN2, a2, b2, c2, event1, event2, event3, event4, dict_manager)
    y1 = res_block0
    block1 = nn.Sequential(*list(new_model[5].children()))[:]
    res_block1 = Block1Server2(res_block0,block1,dict_manager,dict_manager1, event1, event2, event3, event4, a2, b2, c2,t2,c2_res,c2_mul)
    y2 = res_block1
    block2 = nn.Sequential(*list(new_model[6].children()))[:]
    res_block2 = Block2Server2(res_block1,block2,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,256)
    y3 = res_block2
    block3 = nn.Sequential(*list(new_model[7].children()))[:]
    res_block3 = Block2Server2(res_block2,block3,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    y4 = res_block3

    #####浅层特征提取############
    y_concat_1 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)(y1)
    y_concat_12 = torch.cat((y_concat_1, y2), 1)
    y_concat_12_res = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)(y_concat_12)
    y_concat_123 = torch.cat((y_concat_12_res, y3), 1)
    y_concat_123_res = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)(y_concat_123)
    y_concat = torch.cat((y_concat_123_res, y4), 1)
    y_concat = F.adaptive_avg_pool2d(y_concat, (1, 1))

    block4 = nn.Sequential(*list(new_model[8].children()))[:]
    res_block4 = Block3Server2(res_block3,block4,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block5 = nn.Sequential(*list(new_model[9].children()))[:]
    res_block5 = Block3Server2(res_block4,block5,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block6 = nn.Sequential(*list(new_model[10].children()))[:]
    res_block6 = Block3Server2(res_block5,block6,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block7 = nn.Sequential(*list(new_model[11].children()))[:]
    res_block7 = Block3Server2(res_block6,block7,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block8 = nn.Sequential(*list(new_model[12].children()))[:]
    res_block8 = Block3Server2(res_block7,block8,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block9 = nn.Sequential(*list(new_model[13].children()))[:]
    res_block9 = Block3Server2(res_block8,block9,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block10 = nn.Sequential(*list(new_model[14].children()))[:]
    res_block10 = Block3Server2(res_block9,block10,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block11 = nn.Sequential(*list(new_model[15].children()))[:]
    res_block11 = Block3Server2(res_block10,block11,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728)
    block12 = nn.Sequential(*list(new_model[16].children()))[:]
    res_block12 = Block4Server2(res_block11,block12,dict_manager,dict_manager1,event1,event2,event3,event4,a2,b2,c2,t2,c2_res,c2_mul,728,1024)
    #####################最后一层 需要前面的MLFF ##########################

    x = new_model[17](res_block12)
    x = GetBN(new_model[18],1536)(x)
    x = ReluDuofang.reluForServer2(x,dict_manager,event1,event2,event3,event4,a2,b2,c2)
    x = new_model[19](x)
    x = GetBN(new_model[20],2048)(x)
    x = ReluDuofang.reluForServer2(x, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.cat((x,y_concat),1)
    x = x.view(x.size(0),-1)
    x = new_model[21](x)
    x = new_model[22](x)


    #print(type(x))
    #print(res_block1.shape)

    p3.send(x.detach().numpy())
    return

def Server1(new_model,x1,p1,p2,p3,p4,a1,b1,c1,t1,c1_res,c1_mul,event1,event2,event3,event4,dict_manager,dict_manager1):
    Convd1 = new_model[0]
    BN1 = new_model[1]
    Convd2 = new_model[3]
    BN2 = new_model[4]

    print("block0开始")
    res_block0 = Block0Server1(x1,Convd1,BN1,Convd2,BN2,a1,b1,c1,event1,event2,event3,event4,dict_manager)
    y1 = res_block0
    print("block0完成")
    block1 = nn.Sequential(*list(new_model[5].children()))[:]
    res_block1 = Block1Server1(res_block0,block1,dict_manager,dict_manager1, event1, event2, event3, event4, a1, b1, c1,t1,c1_res,c1_mul)
    print("block1完成")
    y2 = res_block1
    block2 = nn.Sequential(*list(new_model[6].children()))[:]
    res_block2 = Block2Server1(res_block1,block2,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,256)
    print("block2完成")
    y3 = res_block2
    block3 = nn.Sequential(*list(new_model[7].children()))[:]
    res_block3 = Block2Server1(res_block2,block3,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block3完成")
    y4 = res_block3

    #####浅层特征提取############
    y_concat_1 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)(y1)
    y_concat_12 = torch.cat((y_concat_1, y2), 1)
    y_concat_12_res = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)(y_concat_12)
    y_concat_123 = torch.cat((y_concat_12_res, y3), 1)
    y_concat_123_res = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)(y_concat_123)
    y_concat = torch.cat((y_concat_123_res, y4), 1)
    y_concat = F.adaptive_avg_pool2d(y_concat, (1, 1))

    block4 = nn.Sequential(*list(new_model[8].children()))[:]
    res_block4 = Block3Server1(res_block3,block4,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block4")
    block5 = nn.Sequential(*list(new_model[9].children()))[:]
    res_block5 = Block3Server1(res_block4,block5,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block5")
    block6 = nn.Sequential(*list(new_model[10].children()))[:]
    res_block6 = Block3Server1(res_block5,block6,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block6")
    block7 = nn.Sequential(*list(new_model[11].children()))[:]
    res_block7 = Block3Server1(res_block6,block7,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block7")
    block8 = nn.Sequential(*list(new_model[12].children()))[:]
    res_block8 = Block3Server1(res_block7,block8,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block8")
    block9 = nn.Sequential(*list(new_model[13].children()))[:]
    res_block9 = Block3Server1(res_block8,block9,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block9")
    block10 = nn.Sequential(*list(new_model[14].children()))[:]
    res_block10 = Block3Server1(res_block9,block10,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("block10")
    block11 = nn.Sequential(*list(new_model[15].children()))[:]
    res_block11 = Block3Server1(res_block10,block11,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728)
    print("bloc11")
    block12 = nn.Sequential(*list(new_model[16].children()))[:]
    res_block12 = Block4Server1(res_block11,block12,dict_manager,dict_manager1,event1,event2,event3,event4,a1,b1,c1,t1,c1_res,c1_mul,728,1024)
    print("bloc12")
    #####################最后一层 需要前面的MLFF ##########################
    x = new_model[17](res_block12)
    x = GetBN(new_model[18],1536)(x)
    x = ReluDuofang.reluForServer1(x,dict_manager,event1,event2,event3,event4,a1,b1,c1)
    x = new_model[19](x)
    x = GetBN(new_model[20],2048)(x)
    x = ReluDuofang.reluForServer1(x, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.cat((x,y_concat),1)
    x = x.view(x.size(0),-1)
    x = new_model[21](x)
    x = new_model[22](x)


    print(x.shape)
    p4.send(x.detach().numpy())
    return



def xce_agg(x1,x2,new_model):
    # 做操作
    p1, p2 = Pipe()
    p3,p4 = Pipe()
    event1 = torch.multiprocessing.Event()
    event2 = torch.multiprocessing.Event()
    event3 = torch.multiprocessing.Event()
    event4 = torch.multiprocessing.Event()
    dict_manager = Manager().dict()
    dict_manager1 = Manager().dict()
    a = randomC.random_c()
    b = randomC.random_c()
    a1 = randomC.random_c()
    a2 = a - a1
    b1 = randomC.random_c()
    b2 = b - b1
    c = a*b
    c1 = randomC.random_c()
    c2 = c - c1

    c1_mul = randomC.random_c()
    c2_mul = randomC.random_c()

    c_mul = c1_mul * c2_mul
    c1_res = randomC.random_c()
    c2_res = c_mul - c1_res

    t = randomC.random_c()
    t1 = randomC.random_c()
    t2 = t - t1

    process1 = Process(target=Server1,args=(new_model,x1,p1,p2,p3,p4,a1,b1,c1,t1,c1_res,c1_mul,event1,event2,event3,event4,dict_manager,dict_manager1))

    process2 = Process(target=Server2,args=(new_model,x2,p1,p2,p3,p4,a2,b2,c2,t2,c2_res,c2_mul,event1,event2,event3,event4,dict_manager,dict_manager1))

    process1.start()
    process2.start()
    process1.join()
    process2.join()
    f1 = p3.recv()
    f2 = p4.recv()

    return f1+f2

# 定义解密函数
def decrypt(ciphertext_image, key):
    ciphertext = np.array(ciphertext_image).flatten()
    binary_ciphertext = np.unpackbits(ciphertext)
    binary_plaintext = np.bitwise_xor(binary_ciphertext, key)
    plaintext = np.packbits(binary_plaintext)
    plaintext_image = Image.fromarray(plaintext.reshape(ciphertext_image.size))
    return plaintext_image


if __name__ == '__main__':
    xgg = xgg_model.xception_agg()
    xgg.load_state_dict(torch.load('', map_location='cpu'))
    # 将这个 new_nodel 传给两个进程
    # 用管道来传递数据
    new_model = nn.Sequential(*list(xgg.children()))[:]

    correct = 0
    total = 0
    list = []
    transform = transforms.Compose([
        #    transforms.RandomResizedCrop(224),
        ##    transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #   transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225]),
    ])

    testData = dsets.ImageFolder('', transform)

    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=5, shuffle=True)

    for images, labels in testLoader:
        #images = images.cuda()
        image_1, image_2 = input.readImg(images)
        ## 解密
        image_1 = decrypt(image_1,key)
        image_2 = decrypt(image_2,key)
        outputs = xce_agg(image_1,image_2,new_model)
        print(outputs)
        _, predicted = torch.max(torch.tensor(outputs).data, 1)
        total += labels.size(0)
        print(labels.size(0))
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100 * correct / total))
        list.append(100 * correct / total)

    for i in list:
        sum += list[i]
    print(sum / len(list))
    #

    # print(image_1.shape)
    # x1 = net(image_1)
    # x2 = net(image_2)
    # print(x1+x2)

    # image = cv2.imread('test.png')
    # image = cv2.resize(image, (256, 256))
    # image = image/255.0
    # image = np.transpose(image, (2, 0, 1))
    # image = np.array(image, np.float32)
    # image_1, image_2 = input.readImg("test.png")
    # # image = torch.unsqueeze(image, dim=0)
    # #image = torch.tensor(image,dtype=torch.float32)
    # #image = torch.unsqueeze(image, dim=0)
    # ProbabilityValue = xce_agg(image_1,image_2,new_model)
    # print(ProbabilityValue)

    # #
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
    # print("block1_3:",block1_3.shape)
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
    # print("x shape:",x.shape)
    # sigmoid1 = ChannelAttention[5](x)
    # print("sigmoid1 shape:", sigmoid1.shape)
    # x = sigmoid1*block1_3
    # print("x2 shape:", x.shape)
    #
    # SpatialAttention = nn.Sequential(*list(block1[4].children()))[:]
    # avg_mean = torch.mean(x,dim=1, keepdim=True)
    # avg_max,_ = torch.max(x, dim=1, keepdim=True)
    # x = torch.cat([avg_mean, avg_max], dim=1)
    # x = SpatialAttention[0](x)
    # x = SpatialAttention[1](x)
    #
    #
    #
    #
    # # print(-6.5986490e+00+6.59865)
    # # print(-15.247357+15.2473545)
    #
    # #print("最后结果 ",sigmoid1.shape)
    # print(x)

    #print(ProbabilityValue)





