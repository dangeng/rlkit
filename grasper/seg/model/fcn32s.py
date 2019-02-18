import numpy as np

import torch
import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator

from .resnet import Res50Dil8

class fcn32s(nn.Module):
    """
    FCN-32s: fully convolutional network with VGG-16 backbone.
    """

    def __init__(self, num_classes, feat_dim=None):
        super().__init__()
        self.num_classes = num_classes
        #self.feat_dim = feat_dim or 4096   # VGG
        self.feat_dim = feat_dim or 2048    # RESNET

        # feature encoder (with ILSVRC pre-training)
        #self.encoder = vgg16(is_caffe=True)    $ VGG
        self.encoder = Res50Dil8()              # RESNET

        # classifier head
        self.head = nn.Conv2d(self.feat_dim, self.num_classes, 1)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

        # bilinear interpolation for upsampling
        #self.decoder = Interpolator(self.num_classes, 32, odd=False)   # VGG
        self.decoder = Interpolator(self.num_classes, 8, odd=False)     # RESNET
        # align output to input: see
        # https://github.com/BVLC/caffe/blob/master/python/caffe/coord_map.py
        #self.encoder[0].padding = (81, 81)     # VGG
        self.crop = 0


    def forward(self, x):
        h, w = x.size()[-2:]
        x = self.encoder(x)
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x
