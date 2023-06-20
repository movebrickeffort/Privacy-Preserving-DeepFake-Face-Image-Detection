import torch.nn as nn
import torch
import torch.quantization
import numpy as np
from PIL import Image
from key import key
# 定义混沌系统的参数
a = 10
b = 8 / 3
r = 28

# 定义初始状态
x0 = 0.1
y0 = 0.2
z0 = 0.3

# 定义迭代次数
n = 100000



# 定义加密函数
def encrypt(image, key):
    plaintext = np.array(image).flatten()
    binary_plaintext = np.unpackbits(plaintext)
    binary_ciphertext = np.bitwise_xor(binary_plaintext, key)
    ciphertext = np.packbits(binary_ciphertext)
    ciphertext_image = Image.fromarray(ciphertext.reshape(image.shape))
    return ciphertext_image



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
    ## 对其进行加密

    image_1 = encrypt(image_1, key)
    image_2 = encrypt(image_2, key)

    image_1 = torch.tensor(image_1,dtype=torch.float32)
    image_2 = torch.tensor(image_2, dtype=torch.float32)
    # image_1 = torch.unsqueeze(image_1, dim=0)
    # image_2 = torch.unsqueeze(image_2, dim=0)


    return image_1, image_2


if __name__ == '__main__':
    print(-1.2433e+01+1.5746)



