import torch
from torch.functional import Tensor
from torchvision import transforms
import torch.nn.functional as F
import math
import random
'''
这个是对原patch进行缩小，参数size小于原patch的size
'''

class Augment:
    def __init__(self,params) -> None:
        self.name=params['type']

    def do(self,data):
        return self.real_do(data)
    
    def real_do(self,data)->Tensor:
        pass

class ShrinkAugment(Augment):
    def __init__(self,params) -> None:
        super(ShrinkAugment,self).__init__(params)
        self.size=params.get("size",3)

    def real_do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        center=int(data.size(2)/2)
        margin=int((self.size-1)/2)
        newdata=torch.zeros(data.size())
        newdata[:,:,center-margin:center+margin+1,center-margin:center+margin+1]=data[:,:,center-margin:center+margin+1,center-margin:center+margin+1]
        
        return newdata

'''
使用高斯核对每个spectrum进行模糊，参数包括kernel_size和sigma_square
在json中：
"type":"Gauss"，
"kernel_size":5
"sigma_sq":2.25
'''
class GaussAugment(Augment):
    def __init__(self,params) -> None:
        super(GaussAugment,self).__init__(params)
        self.kernel_size=params.get("kernel_size",3)
        self.sigma_sq=params.get("sigma_sq",2.25)

    def real_do(self,data):
        # data: batch,channel,patch_size,patch_size
        t=transforms.GaussianBlur(self.kernel_size,self.sigma_sq)
        newdata=t(data)
        return newdata

'''
使用在spectrum维的gaussblur
"type":"SpectralFilter"，
"kernel_size":5
"sigma_sq":2.25
'''
class SpecFilterAugment(Augment):
    def __init__(self,params) -> None:
        super(SpecFilterAugment,self).__init__(params)
        self.kernel_size=params.get("kernel_size",3)
        self.sigma_sq=params.get("sigma_sq",2.25)
        self.margin=self.kernel_size/2
        self.filter=torch.Tensor(self.kernel_size)
        for i in range(self.margin+1):
            self.filter[i]=self.filter[self.kernel_size-1-i]=-1*torch.exp((self.margin-i)*(self.margin-i)/2/self.sigma_sq)/torch.sqrt(2*torch.PI*self.sigma_sq)

    def real_do(self,data):
        # data: batch,channel,patch_size,patch_size
        batch_size=data.size(0)
        channel_num=data.size(1)
        H=data.size(2)
        W=data.size(3)
        data=torch.transpose(data,(0,2,3,1))
        newdata=torch.zeros(data.shape())
        for i in range(batch_size):
            padding_data=torch.zeros(H,W,channel_num+2*self.margin)
            padding_data[:,:,self.margin:self.margin+channel_num+1]=data[i]
            for j in range(H):
                for k in range(W):
                    for l in range(channel_num):
                        newdata[i][j][k][l]=torch.dot(self.filter,padding_data[j][k][l:l+self.kernel_size])
        data=torch.transpose(data,(0,3,1,2))
        newdata=torch.transpose(newdata,(0,3,1,2))
        return newdata

class FlipAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.mirror=params.get('mirror','horizontal')
    
    def real_do(self,data):# b c h w
        if self.mirror=='horizontal':
            return transforms.functional.hflip(data)
        else:
            return transforms.functional.vflip(data)

class RotateAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.angle=params.get('angle',90) # 默认90，也可以是270，逆时针为正

    def real_do(self, data):
        newdata=torch.transpose(data,2,3)
        if self.angle==270:
            return transforms.functional.hflip(newdata)
        else:
            return transforms.functional.vflip(newdata)

class DownSampleAugment(Augment):
    # 降采样
    def __init__(self, params) -> None:
        super().__init__(params)
        self.scale=params.get("scale",2)

    def real_do(self, data):
        x=F.interpolate(data,scale_factor=(1./self.scale,1./self.scale))
        return F.interpolate(x,size=(data.size(2),data.size(3)))

class MaskAugment(Augment):# 3D随机mask,指的是mask大小随机再加left_top点随机
    def __init__(self, params) -> None:
        super().__init__(params)
        self.spc_fac=params

class SameAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)





class BlockMaskAugment(Augment):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.ratio=params['ratio']
        self.layer_idx=0
        self.chosen=params['chosen']

    def generate_mask(self,data):# 这里的data已经进行了降维
        # 按照ratio随机一个mask起始位置
        mask=torch.ones_like(data)
        len=data.size(1)# 获取batch后一维的数据长度
        m=math.ceil(len*self.ratio,)
        if(len(data.size())==2):
            idx=random.randint(0,len-m-1)
            mask[:,idx:idx+m]=0
        elif(len(data.size())==3):
            idx1=random.randint(0,len-m-1)
            idx2=random.randint(0,len-m-1)
            mask[:,idx1:idx1+m,idx2:idx2+m]=0
        return mask
        
    def real_do(self, data) -> Tensor:
        b, s, h, w = data.shape
        margin=int(h/2)
        left=data[:,:,margin,margin]
        left=torch.reshape(left,(b,s))
        left_mask=self.generate_mask(left)
        right=data[:,self.chosen,:,:] # 这个layer怎么来？
        right=torch.unsqueeze(right,dim=1)
        right_mask=self.generate_mask(right)
        return left*left_mask,right*right_mask

def do_augment(params,data):# 增强也有一系列参数呢，比如multiscale的尺寸、mask的大小、Gaussian噪声的参数等
    if params['type']=='shrink':
        return ShrinkAugment(params).do(data)
    if params['type']=='Gauss':
        return GaussAugment(params).do(data)
    if params['type']=='Flip':
        return FlipAugment(params).do(data)
    if params['type']=='Rotate':
        return RotateAugment(params).do(data)
    if params["type"]=='DownSample':
        return DownSampleAugment(params).do(data)
    if params['type'] == 'Same':
        return SameAugment(params).do(data)
    if params['type'] == 'BlockMask':
        return BlockMaskAugment(params).do(data)
