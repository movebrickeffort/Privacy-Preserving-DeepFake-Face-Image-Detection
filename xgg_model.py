
'''
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
'''
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os

# LEARNING_RATE = 0.00001
# EPOCH = 50
# BATCH_SIZE = 32
#
# model_urls = {
#     'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
# }
#
# transform = transforms.Compose([
# #    transforms.RandomResizedCrop(224),
# ##    transforms.RandomHorizontalFlip(),
#       transforms.Resize([256,256]),
#       transforms.ToTensor(),
#
# #   transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                         std=[0.229, 0.224, 0.225]),
# ])
#
# trainData = dsets.ImageFolder('/data1/celeba_df/train/',transform)
# testData = dsets.ImageFolder('/data1/celeba_df/test/',transform)
#
# trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            #注意力

            rep.append(nn.BatchNorm2d(filters))


        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        self.ca = ChannelAttention(out_filters)
        self.sa = SpatialAttention()

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)


        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)
        x = self.ca(x) * x
        x = self.sa(x) * x

        #print('x.shape: ',x.shape)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        #print('skip.shape: ',skip.shape)
        #print(skip)
        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()


        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(3224, 2048)
        self.fc2 = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        y1 = x

        x = self.block1(x)
        y2 = x
        x = self.block2(x)
        y3 = x
        #y3 = concate(y1,y2)
        x = self.block3(x)
        y4 = x
        # print('y1: ', y1.shape)
        # print('y2: ', y2.shape)
        # print('y3: ', y3.shape)
        # print('y4: ', y4.shape)
        y_concat_1 = nn.AvgPool2d((3,3),stride=(2,2),padding=1)(y1)
        # y_concat_2 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=0)(y_concat_1)
        # y_concat_3 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=0)(y_concat_2)
        # print('y_concat.shape: ',y_concat_1.shape)
        # print('y2.shape: ', y2.shape)
        y_concat_12 = torch.cat((y_concat_1,y2),1)
        # print('y_concat_12.shape: ', y_concat_12.shape)
        y_concat_12_res = nn.AvgPool2d((3,3),stride=(2,2),padding=1)(y_concat_12)
        y_concat_123 = torch.cat((y_concat_12_res,y3),1)
        y_concat_123_res = nn.AvgPool2d((3,3),stride=(2,2),padding=1)(y_concat_123)
        # print('y_concat_123_res.shape: ', y_concat_123_res.shape)
        y_concat = torch.cat((y_concat_123_res,y4),1)
        y_concat = F.adaptive_avg_pool2d(y_concat, (1, 1))
        # x_concat_1 = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_1)
        # x_concat_12 = keras.layers.concatenate([x_concat_1, x_2], axis=-1)
        # x_concat_12_res = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_concat_12)
        # x_concat_123 = keras.layers.concatenate([x_concat_12_res, x_3], axis=-1)
        # x_concat_123_res = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_concat_123)
        # x_concat = keras.layers.concatenate([x_concat_123_res, x_4], axis=-1)
        # # x_concat = Conv2D(2048, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_concat)
        #
        # x_concat = GlobalAveragePooling2D()(x_concat)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        print("block12开始")
        x = self.block12(x)
        print("block12完成")

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))

        # print('x.shape: ', x.shape)
        # print('y_concat.shape: ', y_concat.shape)
        x = torch.cat((x, y_concat), 1)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        # print('x_fin.shape: ', x.shape)
        return x



def xception_agg(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)

    return model

# def train():
#     xgg = xception_agg()
#     xgg.load_state_dict(torch.load('/data1/lx/code/model/xgg_celebadf/0cnn.pkl'))
#     use_gpu = torch.cuda.is_available()
#     if use_gpu:
#         xgg.cuda()
#
#     # Loss, Optimizer & Scheduler
#     cost = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(xgg.parameters(), lr=LEARNING_RATE)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#
#     # Train the model
#     for epoch in range(EPOCH):
#
#         avg_loss = 0
#         cnt = 0
#         for images, labels in trainLoader:
#             if use_gpu:
#                 images = images.cuda()
#                 labels = labels.cuda()
#
#             # Forward + Backward + Optimize
#             optimizer.zero_grad()
#             outputs = xgg(images)
#             loss = cost(outputs, labels)
#             avg_loss += loss.data
#             cnt += 1
#             print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
#             loss.backward()
#             optimizer.step()
#         scheduler.step(avg_loss)
#         torch.save(xgg.state_dict(), '/data1/lx/code/model/xgg_celebadf/'+str(epoch+1) + 'cnn.pkl')
#
#     # Test the model
#         xgg.eval()
#         correct = 0
#         total = 0
#
#         for images, labels in testLoader:
#             images = images.cuda()
#             outputs = xgg(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted.cpu() == labels).sum()
#             print(predicted, labels, correct, total)
#             print("avg acc: %f" % (100 * correct / total))
#
#     # Save the Trained Model
#     torch.save(xgg.state_dict(), 'cnn.pkl')
#
# if __name__ == '__main__':
#     train()