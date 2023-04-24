import torch
from torch import nn
from models import conv2d
from models import conv1d
import collections
'''
visual部分的选取，只选择一个band，在每个epoch中从channel中随机选取
mask：在训练过程中随机选取v或s部分，以比率r的方式进行mask，另一个未被选中的mask设为1
三个块先后生成128、256、512的feature map
'''

class SpectralModule(torch.nn.Module):# 光谱维，应该是一维的
    def __init__(self,params):
        super().__init__()
        self.out_d=params['net'].get('mlp_head',64)
        self.layer1=nn.Sequential(collections.OrderedDict([
            ('conv',nn.Conv1d(1,64,3,1,1))
            ,('relu',nn.ReLU())
        ]))
        self.b1=conv1d.Conv1dBlock(64,128)
        self.b2=conv1d.Conv1dBlock(128,256)
        self.b3=conv1d.Conv1dBlock(256,512)

        self.mlp=nn.Sequential(collections.OrderedDict([
            ('flat',nn.Flatten())
            ,('fc1',nn.LazyLinear(self.out_d*2))
            ,('relu',nn.ReLU())
            ,('fc2',nn.Linear(self.out_d*2,self.out_d))
            ,('relu',nn.ReLU())
        ]))
    
    def forward(self,x):
        x=torch.unsqueeze(x,dim=1)
        h=self.layer1(x)
        h=self.b1(h)
        h=self.b2(h)
        h=self.b3(h)
        return self.mlp(h)
    
class VisualModule(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.out_d=params['net'].get('mlp_head',64)
        self.layer1=nn.Sequential(collections.OrderedDict([
            ('conv',nn.Conv2d(1,64,3,1,1))
            ,('relu',nn.ReLU())
        ]))
        self.b1=conv2d.Conv2dBlock(64,128)
        self.b2=conv2d.Conv2dBlock(128,256)
        self.b3=conv2d.Conv2dBlock(256,512)

        self.mlp=nn.Sequential(collections.OrderedDict([
            ('flat',nn.Flatten())
            ,('fc1',nn.LazyLinear(self.out_d*2))
            ,('relu',nn.ReLU())
            ,('fc2',nn.Linear(self.out_d*2,self.out_d))
            ,('relu',nn.ReLU())
        ]))

    def forward(self,x):# 
        x=torch.unsqueeze(x,dim=1)
        h=self.layer1(x)
        h=self.b1(h)
        h=self.b2(h)
        h=self.b3(h)
        return self.mlp(h)
    
class BaseEncoder(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.fs=SpectralModule(params)
        self.fv=VisualModule(params)

    def forward(self,s,v):# s是spectral，[batch,C]; v是visual,[batch H H]
        h1=self.fs(s)
        h2=self.fv(v)
        return h1,h2
    
class XDCL(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.num_class=params['data']['num_classes']
        self.mlp_head=params['net']['mlp_head']
        self.backbone=BaseEncoder(params)
        self.classifier=nn.Sequential(collections.OrderedDict([
            ('fc',nn.Linear(2*self.mlp_head,self.num_class))
            ,('relu',nn.ReLU())
        ]))

    def forward(self,s,v):
        h1,h2=self.backbone(s,v)
        h3=torch.cat([h1,h2],dim=1)
        return self.classifier(h3),h1,h2