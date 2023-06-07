import torch
import math
import itertools
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable, Function


class gaussian_noise(nn.Module):
    def __init__(self, mean=0, std=0.15):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        noise = input.data.new(input.size()).normal_(self.mean, self.std)
        return input + noise


class Pi_model(nn.Module):
    def __init__(self, input_chann_num=3, num_classes=10):
        super().__init__()
        self.gauss = gaussian_noise()
        self.conv1a = weight_norm(
            nn.Conv2d(
                in_channels=input_chann_num,
                out_channels=128,
                kernel_size=3,
                padding="same",
            )
        )
        self.conv1b = weight_norm(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
        )
        self.conv2a = weight_norm(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        )
        self.conv2b = weight_norm(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same")
        )

        self.conv3a = weight_norm(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="valid")
        )
        self.conv3b = weight_norm(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        )
        self.conv3c = weight_norm(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, batch):
        x = batch.imgs
        x = self.gauss(x)  # gaussian noise
        x = self.lrelu(self.conv1a(x))  # conv1a
        x = self.lrelu(self.conv1b(x))  # conv1b
        x = self.lrelu(self.conv1b(x))  # conv1c
        x = self.maxpool(x)  # pool1
        x = self.dropout(x)  # drop1
        x = self.lrelu(self.conv2a(x))  # conv2a
        x = self.lrelu(self.conv2b(x))  # conv2b
        x = self.lrelu(self.conv2b(x))  # conv2c
        x = self.maxpool(x)  # pool2
        x = self.dropout(x)  # drop2
        x = self.lrelu(self.conv3a(x))  # conv3a
        x = self.lrelu(self.conv3b(x))  # conv3b
        x = self.lrelu(self.conv3c(x))  # conv3c
        x = self.gap(x)  # pool3
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)  # dense
        return x


class Residual(nn.Module):  # @save
    def __init__(self, in_chann_num, out_chann_num, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chann_num, out_chann_num, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(out_chann_num, out_chann_num, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_chann_num, out_chann_num, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_chann_num)
        self.bn2 = nn.BatchNorm2d(out_chann_num)

    def forward(self, X):
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(in_chann_num, out_chann_num, residual_num, first_block=False):
    layers = []
    for i in range(residual_num):
        if i == 0 and not first_block:
            layers.append(
                Residual(in_chann_num, out_chann_num, use_1x1conv=True, strides=2)
            )
        else:
            layers.append(Residual(out_chann_num, out_chann_num))
    return layers


class ResNet18(nn.Module):
    def __init__(self, input_chann_num, output_chann_num):
        super().__init__()
        p1 = nn.Sequential(
            nn.Conv2d(input_chann_num, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        p2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        p3 = nn.Sequential(nn.Dropout(p=0.5), *resnet_block(64, 128, 2))
        p4 = nn.Sequential(nn.Dropout(p=0.5), *resnet_block(128, 256, 2))
        p5 = nn.Sequential(nn.Dropout(p=0.5), *resnet_block(256, 512, 2))
        p6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, output_chann_num)
        )
        self.net = nn.Sequential(p1, p2, p3, p4, p5, p6)

    def forward(self, batch):
        x = batch.imgs
        return self.net(x)


def cifar_shakeshake26(input_chann_num, output_chann_num):
    model = ResNet32x32(
        input_chann_num,
        ShakeShakeBlock,
        layers=[4, 4, 4],
        channels=output_chann_num,
        downsample="shift_conv",
    )
    return model


class ResNet32x32(nn.Module):
    def __init__(
        self,
        input_size,
        block,
        layers,
        channels,
        groups=1,
        num_classes=10,
        downsample="basic",
    ):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            input_size, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(block.out_channels(channels * 4, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == "basic" or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        block.out_channels(planes, groups),
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == "shift_conv":
                downsample = ShiftConvDownsample(
                    in_channels=self.inplanes,
                    out_channels=block.out_channels(planes, groups),
                )
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, batch):
        x = batch.imgs
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        residual = x
        a = F.relu(x, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + a


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=2 * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2], x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
