#!/usr/bin/env python

import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class mulMod3D_MIL(nn.Module):

    def __init__(self, input_channels, model_name='resnet18', heatmaps=False ):
        super(mulMod3D_MIL, self).__init__()
        expansion = 1

        self.heatmaps = heatmaps
        self.model_name = model_name

        # resnet model is until average pooling
        if self.model_name == 'resnet18' :
            resnet = models.resnet18(pretrained=True)

        # change the default channel
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(resnet.children())[0:-1])

        #gated attention mechanism
        if self.model_name == 'resnet18':
            self.L = 512
            self.D = 256
            self.K = 1

        #attention-based pooling
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights =  nn.Linear(self.D, self.K)

        self.regressor = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.LeakyReLU(0.3, inplace=True)
        )


    def forward(self, x):
        b,s, c, w, h = x.shape

        x = x.view(b*s, c, w, h)

        H = self.backbone(x)

        batch_slices, channel, width, height = H.shape

        H = H.view(batch_slices, channel*width*height) ## [155, 512]!!

        #attention-based pooling
        A_V = self.attention_V(H)  

        A_U = self.attention_U(H)  

        A_weights = self.attention_weights(A_V * A_U)  

        n, k = A_weights.shape 

        A = F.softmax(A_weights, dim=0)

        heatmap_weights = A

        A = torch.transpose(A, 1, 0)

        M = torch.mm(A, H) 

        Y_pred = self.regressor(M) 

        if self.heatmaps:
            return Y_pred, heatmap_weights
        else:
            return Y_pred



