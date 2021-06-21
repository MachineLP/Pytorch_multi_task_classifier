# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QDNet Model effnet
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------
'''



import torch
import torch.nn as nn
import torchvision
from qdnet.conf.constant import Constant
from qdnet.models.metric_strategy import Swish_module, ArcMarginProduct_subcenter, ArcFaceLossAdaptiveMargin

class Resnet(nn.Module):
    '''
    '''
    def __init__(self, enet_type, out_dim, drop_nums=1, pretrained=False, metric_strategy=False):
        super(Resnet, self).__init__()
        if enet_type == Constant.RESNET_LIST[0]:
            self.model = torchvision.models.resnet18(pretrained=pretrained)
            self.model.fc = torch.nn.Sequential( torch.nn.Linear( in_features=512, out_features=out_dim ) )
        if enet_type == Constant.RESNET_LIST[1]:
            self.model = torchvision.models.resnet34(pretrained=pretrained)
            self.model.fc = torch.nn.Sequential( torch.nn.Linear( in_features=1024, out_features=out_dim ) )
        if enet_type == Constant.RESNET_LIST[2]:
            self.model = torchvision.models.resnet50(pretrained=pretrained)
            self.model.fc = torch.nn.Sequential( torch.nn.Linear( in_features=2048, out_features=out_dim ) )

   
    def forward(self, x):
        out = self.model(x)  
        return out

