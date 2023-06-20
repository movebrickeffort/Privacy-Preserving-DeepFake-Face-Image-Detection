import numpy as np

# 定义全局变量
key = None

# 定义混沌系统的参数
a = 10
b = 8 / 3
r = 28

# 定义初始状态
x0 = 0.1
y0 = 0.2
z0 = 0.3

# 定义迭代次数
n = 10000

# 定义密钥生成函数
def key_gen():
    global key
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    x[0], y[0], z[0] = x0, y0, z0
    for i in range(1, n):
        x[i] = y[i-1] + a*(y[i-1] - x[i-1])
        y[i] = x[i-1] * (r - z[i-1]) - y[i-1]
        z[i] = x[i-1] * y[i-1] - b*z[i-1]
    key = np.mod(np.round(z*1000), 256).astype(int)