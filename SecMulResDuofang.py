import torch

def secMulResServer2(x2_mul, c2_mul, c2_res,dict_manager,event1,event2):
    #print("secMulRes2")
    a2 = x2_mul / c2_mul

    #p2.send(a2)
    dict_manager.update({'a2':a2.detach()})
    event1.set()
    event2.wait()
    #a1 = p2.recv()
    a1 = dict_manager['a1']
    #a1 = torch.tensor(a1, dtype=torch.float32)

    a = a1 * a2

    x2_res = a * c2_res
    #print(x2_res)

    return x2_res

def secMulResServer1(x1_mul,c1_mul,c1_res,dict_manager,event1,event2):
    #print("secMulRes1")
    a1 = x1_mul/c1_mul

    #p1.send(a1)
    dict_manager.update({'a1':a1.detach()})
    event2.set()
    event1.wait()
    #a2 = p1.recv()
    a2 = dict_manager['a2']
    #2 = torch.tensor(a2,dtype=torch.float32)

    a = a1*a2

    x1_res = a*c1_res

    return x1_res