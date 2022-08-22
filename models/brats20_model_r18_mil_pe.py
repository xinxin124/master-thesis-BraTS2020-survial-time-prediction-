#!/usr/bin/env python

import torch
from torch import nn
#from resnet import resnet18
import torchvision.models as models
import torch.nn.functional as F
import math


class mul3D_MIL_pe(nn.Module):

    def __init__(self, input_channels, model_name='resnet18', heatmaps=False, device="cuda:0" ):
        super(mul3D_MIL_pe, self).__init__()
        expansion = 1

        self.heatmaps = heatmaps
        self.model_name = model_name

        # resnet model is until average pooling
        if self.model_name == 'resnet18' :
            resnet = models.resnet18(pretrained=True)
        elif self.model_name == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif self.model_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif self.model_name == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        elif self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=True)


        # change the default channel
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(resnet.children())[0:-1])

        #gated attention mechanism
        if self.model_name == 'resnet18' or self.model_name == 'resnet34':
            self.L = 512
            self.D = 256
            self.K = 1
        elif self.model_name == 'resnet50' or self.model_name == 'resnet101' or self.model_name == 'resnet152':
            self.L = 2048
            self.D = 1024
            self.K = 1

        #positional embedding
        self.pe = self.positionalencoding1d(self.L, 155).to(device)

        #attention based pooling
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights =  nn.Linear(self.D, self.K)
            #not clear  dimention  of attention weights?

        self.regressor = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.LeakyReLU(0.3, inplace=True)
        )


        # dropout layer
        # self.dropout_layer = nn.Dropout(p=0.2, inplace=True)

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

    # pe = torch.Size([155, 512])


    def forward(self, x):
        # b, s, c, w, h !!
        b,s, c, w, h = x.shape
        x = x.view(b*s, c, w, h) #[160*155, 1, 160, 160]

        Ha = self.backbone(x)

        batch_slices, channel, width, height = Ha.shape

        #positional encoding
        Ha = Ha.view(batch_slices, channel*width*height) ## [155, 512*1*1]
        H = torch.add(Ha, self.pe) # [155,512]

        A_V = self.attention_V(H)  # NxD [155, 256]

        A_U = self.attention_U(H)  # NxD [155, 256]

        A_weights = self.attention_weights(A_V * A_U)  ## NxK [155,1]

        #A_weights = torch.transpose(A_weights, 1,0)  #KxN  [1, 620]

        n, k = A_weights.shape #n is batch size, k is slices [155, 1]

        A = F.softmax(A_weights, dim=0)  # softmax over N # 3d heatmap, get import slices [155,1]

        heatmap_weights = A #[155, 1]
        sum_weights_after_softmax = sum(A) #[1.0000]

        A = torch.transpose(A, 1, 0) #[1, 155]

        #H = torch.transpose(H, 1, 0)
        M = torch.mm(A, H)  # KxL  [1, 155] %*% [155, 512] = [1, 512]

        # using dropout layer
        #M = self.dropout_layer(M_0)

        Y_pred = self.regressor(M)  #fully connected layer  [1,1]

        if self.heatmaps:
            return Y_pred, heatmap_weights
        else:
            return Y_pred


if  __name__ == "__main__":

    #model = mulMod3D_MIL(1).cuda()
    model = mul3D_MIL_pe(1, 'resnet18')
    #summary(model)

    print(model.eval())





