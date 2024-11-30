import torch
from Constant import *

class NNv1(torch.nn.Module):
    def __init__(self, num_cls=cls_num):
        super(NNv1, self).__init__()
        self.fcs = torch.nn.Sequential(
            # [b, 64] -> [b, 128] -> [b, num_cls]
            torch.nn.Linear(max_length, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(128, num_cls),
        )

    def forward(self, x):
        x = self.fcs(x)

        return x

class NNv2(torch.nn.Module):
    def __init__(self, num_cls=cls_num):
        super(NNv2, self).__init__()
        self.fcs = torch.nn.Sequential(
            # [b, 64] -> [b, 128] -> [b, 128] -> [b, num_cls]
            torch.nn.Linear(max_length, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(128, num_cls),
        )

    def forward(self, x):
        x = self.fcs(x)

        return x

# CNN like VGG16
class conv_bn_relu(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(conv_bn_relu, self).__init__()
        self.convs_bn_relu = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding=2),
            torch.nn.BatchNorm2d(c_out),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.convs_bn_relu(x)

        return x

class CNN(torch.nn.Module):
    def __init__(self, num_cls=sum(num for num in word_count)):
        super(CNN, self).__init__()
        self.convs = torch.nn.Sequential(
            # [b, 1, 28, 28]
            conv_bn_relu(1, 64), # [b, 64, 30, 30]
            conv_bn_relu(64, 64), # [b, 64, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 64, 16, 16]

            conv_bn_relu(64, 128), # [b, 128, 18, 18]
            conv_bn_relu(128, 128), # [b, 128, 20, 20]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 128, 10, 10]

            conv_bn_relu(128, 256), # [b, 256, 12, 12]
            conv_bn_relu(256, 256), # [b, 256, 14, 14]
            conv_bn_relu(256, 256), # [b, 256, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 256, 8, 8]

            conv_bn_relu(256, 512), # [b, 512, 10, 10]
            conv_bn_relu(512, 512), # [b, 512, 12, 12]
            conv_bn_relu(512, 512), # [b, 512, 14, 14]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 512, 7, 7]

            conv_bn_relu(512, 512), # [b, 512, 9, 9]
            conv_bn_relu(512, 512), # [b, 512, 11, 11]
            conv_bn_relu(512, 512), # [b, 512, 13, 13]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 512, 6, 6]
        )

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(512 * 6 * 6, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            torch.nn.Linear(4096, num_cls),
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.reshape(x, (-1, 512 * 6 * 6))
        x = self.fcs(x)

        return x

