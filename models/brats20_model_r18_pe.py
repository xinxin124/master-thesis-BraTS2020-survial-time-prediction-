#!/usr/bin/env python

import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import math 


class r18PE(nn.Module):

    def __init__(self, input_channels=1, device="cuda:0"):
        super(r18PE, self).__init__()
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
        # positional embedding
        self.pe = self.positionalencoding1d(512, 155).to(device)

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def forward(self, x):
        b,s, c, w, h = x.shape
        x = x.view(b*s, c, w, h)

        H = self.backbone(x)

        batch_slices, channel, width, height = H.shape

        Ha = H.view(batch_slices, channel*width*height) 

        H = torch.add(Ha, self.pe)  ## pe = torch.Size([155, 512])

        Y = self.regressor(H)  

        Y_t = torch.transpose(Y, 1, 0) #[1,155]

        Y_pred = self.predictor(Y_t)

        return Y_pred



