import torch.nn as nn
import torch
import torch.quantization
import numpy as np
from Crypto.Cipher import
from pybloomfilter import BloomFilter

# 加密密钥，必须是长度为 16、24 或 32 的字符串
key = '0123456789abcdef'



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
    image_1 = encrypt_image(image_1)
    image_2 = encrypt_image(image_2)

    image_1 = torch.tensor(image_1,dtype=torch.float32)
    image_2 = torch.tensor(image_2, dtype=torch.float32)
    # image_1 = torch.unsqueeze(image_1, dim=0)
    # image_2 = torch.unsqueeze(image_2, dim=0)


    return image_1, image_2

def encrypt_image(image):
    # 将图像分解并加密每个通道的分解结果
    encrypted_channels = []
    for c in range(3):
        # 创建一个 AES 加密器
        cipher = AES.new(key.encode(), AES.MODE_ECB)
        # 将当前通道的分解结果转换为 bytes 类型
        data = bytes(np.ndarray.flatten(image[:,:,c]))
        # 对分解结果进行加密
        encrypted_data = cipher.encrypt(data)
        # 将加密后的结果保存到列表中
        encrypted_channels.append(encrypted_data)
    # 将加密后的结果合并成一个加密后的彩色图像
    encrypted_image = np.zeros(image.shape, np.uint8)
    for c in range(3):
        # 获取图像的宽和高
        h, w = c.shape
        data = ''.join([str(c[i][j]) for i in range(h) for j in range(w)])
        # 创建一个 Bloom Filter，用于保存图像的哈希值
        bf = BloomFilter(capacity=1000000, error_rate=0.001)
        # 将图像的哈希值添加到 Bloom Filter 中
        bf.add(data)
        # 将 Bloom Filter 转换为 bytes 类型
        bf_bytes = bf.tobytes()
        # 创建一个 AES 解密器
        cipher = AES.new(key.encode(), AES.MODE_ECB)
        # 对当前通道的加密结果进行解密
        decrypted_data = cipher.decrypt(bf_bytes)
        # 将解密后的结果转换为 numpy 数组
        decrypted_array = np.frombuffer(decrypted_data, np.uint8)
        # 将解密后的结果重构为原始图像的形状
        decrypted_channel = np.reshape(decrypted_array, (image.shape[0], image.shape[1]))
        # 将解密后的结果赋值给加密后的图像的对应通道
        encrypted_image[:,:,c] = decrypted_channel
    return encrypted_image

if __name__ == '__main__':
    print(-1.2433e+01+1.5746)



