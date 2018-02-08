# freda (todo) :

import numpy as np
import torch
from torch import nn


def conv(in_channels,
         out_channels,
         kernel_size=3,
         stride=1,
         bias=True,
         with_bn=True,
         with_relu=True):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True)
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if with_relu:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*tuple(layers))


def deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True), nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_channels):
    return nn.Conv2d(
        in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i, j, :, :] = torch.from_numpy(bilinear)


def save_grad(grads, name):

    def hook(grad):
        grads[name] = grad

    return hook


'''
def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook
import torch
from channelnorm_package.modules.channelnorm import ChannelNorm 
model = ChannelNorm().cuda()
grads = {}
a = 100*torch.autograd.Variable(torch.randn((1,3,5,5)).cuda(), requires_grad=True)
a.register_hook(save_grad(grads, 'a'))
b = model(a)
y = torch.mean(b)
y.backward()

'''
