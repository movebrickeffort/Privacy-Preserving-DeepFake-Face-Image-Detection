import math
import SecMulResDuofang
import torch




def expServer1(x1_mul,c1_mul,c1_res,dict_manager,event1,event2):
    x = torch.exp(x1_mul)
    x1_res = SecMulResDuofang.secMulResServer1(x,c1_mul,c1_res,dict_manager,event1,event2)

    #print("x1_res:", x1_res)
    #print("server1_exp 完成")
    #print("exp:",type(x1_res))
    #print('x1_res:',x1_res)

    return x1_res



def expServer2(x2_mul,c2_mul,c2_res,dict_manager,event1,event2):
    x = torch.exp(x2_mul)
    x2_res = SecMulResDuofang.secMulResServer2(x, c2_mul, c2_res, dict_manager,event1,event2)

    #print("x2_res:", x2_res)
    #print("server2_exp 完成")
    #print("exp",type(x2_res))
    #print('x2_res:', x2_res)
    return x2_res
