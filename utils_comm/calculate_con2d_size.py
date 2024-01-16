import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
from icecream import ic
import os, sys, shutil
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
sys.path.append(os.path.abspath('.'))


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def cond2d_size(input_size, kernel_size, stride, padding=(0, 0), dilation=(1, 1)):
    """
    docstring
    """
    h_in, w_in = pair(input_size)
    kernel_size_h, kernel_size_w = pair(kernel_size)
    stride_h, stride_w = pair(stride)
    padding_h, padding_w = pair(padding)
    dilation_h, dilation_w = dilation
    h_out = (h_in + 2 * padding_h - dilation_h *(kernel_size_h -1) -1) // stride_h + 1
    w_out = (w_in + 2 * padding_w - dilation_w *(kernel_size_w -1) -1) // stride_w + 1
    return (h_out, w_out)

ks = 7
padding = 3
# padding = int((ks-1)/2)

ic(cond2d_size(50, ks, 1, padding))
ic(cond2d_size(448, 7, 2, 3))
ic(cond2d_size(224, 7, 2, 3))
ic(cond2d_size(448, 3, 2, 1))
ic(cond2d_size(112, 3, 1, 1))
ic(cond2d_size(112, 1, 1, 0))
ic(cond2d_size(112, 2, 2))
ic(cond2d_size(224, 7, 4, 2))
ic(cond2d_size(32, (3, 2) , 2, 0))
ic(cond2d_size((60, 300), (5, 300) , 1, 0))
ic(cond2d_size((58, 3), (1, 4) , 2, 1))
""" 
ic| cond2d_size(448, 7, 2, 3): (224, 224)
ic| cond2d_size(224, 7, 2, 3): (112, 112)
ic| cond2d_size(448, 3, 2, 1): (224, 224)
ic| cond2d_size(112, 3, 1, 1): (112, 112)
ic| cond2d_size(112, 1, 1, 0): (112, 112)
ic| cond2d_size(112, 2, 2): (56, 56)
ic| cond2d_size(224, 7, 4, 2): (56, 56)
 """

# pool of square window of size=3, stride=2
m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((3, 2), 2)
input = torch.randn(20, 16, 50, 32)  # torch.Size([20, 16, 24, 16])
output = m(input)
ic(output.size())

ic('ConvTranspose2d')
# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)
output = m(input)
ic(output.size())
# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
ic(h.size())
# torch.Size([1, 16, 6, 6])
output = upsample(h, output_size=input.size())
# output = upsample(h)
ic(output.size())
# torch.Size([1, 16, 12, 12])

# img_file = 'data/1ant.jpg'
# a = cv2.imread(img_file)
# a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
# print(a.shape)
# b = a.transpose(2, 0, 1)
# print(b.shape)
# print()