import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

conv = nn.Sequential(
    #------------------------------------------#
    #   将RGB+Depth四通道通过1x1卷积降维至三通道
    #------------------------------------------#
    nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, padding=0, bias=False),
    nn.BatchNorm2d(3),
    nn.ReLU(inplace=True)
)

rgb   = np.array(Image.open(os.path.join('Images', '001.jpg')))
depth = np.array(Image.open(os.path.join('Depths', '001.jpg'))).reshape([1200, 1920, 1])

mix = np.concatenate([rgb, depth], axis=2)
mix = mix.transpose([2, 0, 1]).astype(np.float32) / 255.0
mix = torch.tensor(mix)
mix = mix.unsqueeze(0)

mix = conv(mix)
mix = mix.squeeze(0)
