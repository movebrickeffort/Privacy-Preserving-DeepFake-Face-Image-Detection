import random
import torch

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

def generate_share2(Conv,a,b,c,t,c_res,c_mul):
    A = torch.ones_like(Conv) * a
    B = torch.ones_like(Conv) * b
    C = torch.ones_like(Conv) * c
    T = torch.ones_like(Conv) * t
    C_res = torch.ones_like(Conv) * c_res
    C_mul = torch.ones_like(Conv) * c_mul

    return A,B,C,T,C_res,C_mul