import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50

from .resnet import Resnet10CIFAR

from .slim_resnet import SlimResNet18

import copy

def getmodel(arch):
    
    #backbone = resnet18()
    
    if arch == "resnet18-cifar":
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        return backbone, 512  
    elif arch == "resnet18-imagenet":
        backbone = resnet18()    
        backbone.fc = nn.Identity()
        return backbone, 512
    elif arch == "resnet18-tinyimagenet":
        backbone = resnet18()    
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Identity()
        return backbone, 512
    elif arch == "slim-resnet18":
        backbone = SlimResNet18(10)
        features_dim = backbone.linear.in_features
        print('slim Resnet18 features dim:', features_dim)
        backbone.linear = nn.Identity()
        return backbone, features_dim
    else:
        raise NameError("{} not found in network architecture".format(arch))
  

class encoderEMP(nn.Module): 
     def __init__(self,z_dim=1024,hidden_dim=4096, norm_p=2, arch = "resnet18-cifar"):
        super().__init__()

        backbone, feature_dim = getmodel(arch)
        self.backbone = backbone
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim,hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU()
                                        )
        self.projection = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,z_dim))
        
          
     def forward(self, x):
         
        enc_feature = self.backbone(x)
        feature = self.pre_feature(enc_feature)
        z = F.normalize(self.projection(feature),p=self.norm_p)

        
        # return z, enc_feature
        return feature, z
    


class encoderSimSiam(nn.Module): 
     def __init__(self,z_dim=1024, pred_hidden_dim=512, norm_p=2, arch = "resnet18-cifar"):
        super().__init__()

        backbone, feature_dim = getmodel(arch)
        self.backbone = backbone
        self.norm_p = norm_p
        self.projection = nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(feature_dim,z_dim),
                                        nn.BatchNorm1d(z_dim, affine=False)) # output layer
        self.projection[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.predictor = nn.Sequential(nn.Linear(z_dim, pred_hidden_dim, bias=False),
                                nn.BatchNorm1d(pred_hidden_dim),
                                nn.ReLU(inplace=True), # hidden layer
                                nn.Linear(pred_hidden_dim, z_dim)) # output layer

          
     def forward(self, x, is_test = False):
         
        feature = self.backbone(x)
        z = self.projection(feature)
        p = self.predictor(z)

        if is_test:
            return z, feature
        else:
            return feature, z, p
        

class encoderEMPAEP(nn.Module): 
     def __init__(self,z_dim=1024,hidden_dim=4096, norm_p=2, arch = "resnet18-cifar", aligner_dim = 512):
        super().__init__()

        backbone, feature_dim = getmodel(arch)
        self.backbone = backbone
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim,hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU()
                                        )
        self.projection = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,z_dim))

        self.momentum_backbone = copy.deepcopy(backbone)
        self.momentum_prefeature = copy.deepcopy(self.pre_feature)
        self.momentum_projection = copy.deepcopy(self.projection)
        self.momentum_backbone.requires_grad_(False)
        self.momentum_prefeature.requires_grad_(False)
        self.momentum_projection.requires_grad_(False)

        self.alignment_head = nn.Sequential(nn.Linear(z_dim, aligner_dim, bias=False),
                                                nn.BatchNorm1d(aligner_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(aligner_dim, z_dim))

        
          
     def forward(self, x):
         
        enc_feature = self.backbone(x)
        feature = self.pre_feature(enc_feature)
        z = F.normalize(self.projection(feature),p=self.norm_p)

        
        return z, enc_feature
    
   
    