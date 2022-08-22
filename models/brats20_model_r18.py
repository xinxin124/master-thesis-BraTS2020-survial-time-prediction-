#!/usr/bin/env python

import torch
from torch import nn
#from resnet import resnet18
import torchvision.models as models
import torch.nn.functional as F


class mul3D_r18(nn.Module):

    def __init__(self, input_channels=1):
        super(mul3D_r18, self).__init__()
        expansion = 1

        # resnet model is until average pooling
        resnet = models.resnet18(pretrained=True)

        # change the default channel
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(resnet.children())[0:-1])

        self.regressor = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
            )

        self.predictor = nn.Sequential(
            nn.Linear(155, 1),
            nn.LeakyReLU(0.3, inplace=True)
        )


    def forward(self, x):
        b,s, c, w, h = x.shape

        x = x.view(b*s, c, w, h)

        H = self.backbone(x)

        batch_slices, channel, width, height = H.shape

        H = H.view(batch_slices, channel*width*height) ## [155, 512]

        Y = self.regressor(H)  # [155, 1]

        Y_t = torch.transpose(Y, 1, 0) # [1,155]
        
        Y_pred = self.predictor(Y_t)

        return Y_pred


