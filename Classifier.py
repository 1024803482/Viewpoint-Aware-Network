from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import einsum
import torchvision


class ResNet18(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet34(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet50(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNeXt50(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=8,
                 pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        if in_channels == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                          padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    inputs = torch.randn(1, 3, 352, 352)
    model = ResNet18(in_channels=3,
                     num_classes=2,
                     pretrained=False)
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    res = model(inputs)
    print("input shape: {}\noutput shape: {}".format(inputs.shape, res.shape))
