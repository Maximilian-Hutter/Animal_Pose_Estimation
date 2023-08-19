import torch.nn as nn
import torch
import torch.nn.functional as F
from params import hparams

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self,x):
        batch, channels, height, width = x.size()
        assert (channels % self.groups == 0)
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(batch, channels, height, width)
        return out

class CSDLKCB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride = 1, pad = 4, group = 4,cuda=hparams["gpu_mode"]):
        super(CSDLKCB, self).__init__()

        self.use_cuda = cuda
        
        self.pad = nn.ReflectionPad2d((pad, pad, pad, pad))
        self.conv1 = nn.Conv2d(in_feat, in_feat, 1)
        self.shuffle = ChannelShuffle(groups=group)
        self.dwconv = nn.Conv2d(in_feat, in_feat, kernel,stride, groups=in_feat)
        self.dwdconv = nn.Conv2d(in_feat, in_feat, kernel, stride, dilation=kernel, groups=in_feat)
        self.conv2 = nn.Conv2d(in_feat, out_feat, kernel_size=1)

        if cuda:
            self.pad = self.pad.cuda()
            self.conv1 = self.conv1.cuda()
            self.shuffle = self.shuffle.cuda()
            self.dwconv = self.dwconv.cuda()
            self.dwdconv = self.dwdconv.cuda()
            self.conv2 = self.conv2.cuda()
        
    def forward(self,x):
        
        if x.size().__len__() == 3:
            x = x.unsqueeze(dim=0)

        if self.use_cuda:
            x = x.cuda()

        x = self.pad(x)

        x = self.conv1(x)

        x = self.shuffle(x)

        x = self.dwconv(x)

        x = self.dwdconv(x)

        x = self.conv2(x)

        return x
    
class ConvBlock(nn.Module):
    def __init__(self,feat_in, feat_out,activation = "LeakyReLU",kernel=3,stride=1,pad=1,cuda=hparams["gpu_mode"]):
        super().__init__()

        self.conv = nn.Conv2d(feat_in,feat_out,kernel_size=kernel,stride=stride,padding=pad)
        if activation == "ReLU":
            self.activ = nn.ReLU()
        elif activation == "ReLU6":
            self.activ = nn.ReLU6()
        elif activation == "SELU":
            self.activ = nn.SELU()
        elif activation == "RReLU":
            self.activ = nn.RReLU()
        elif activation == "ELU":
            self.activ = nn.ELU()
        elif activation == "LeakyReLU":
            self.activ = nn.LeakyReLU()
        elif activation == "Sigmoid":
            self.activ = nn.Sigmoid()
        elif activation == "SiLU":
            self.activ = nn.SiLU()  

        if cuda:
            self.conv = self.conv.cuda()
            self.activ = self.activ.cuda()

    def forward(self,x):

        #print(x.type())
        # turns float to half float for some reason
        x = self.conv(x)
        #print(x.type())
        out = self.activ(x)

        return out

class CEFN(nn.Module):
    def __init__(self,feat,pool_kernel,pool_stride, shape):
        super(CEFN, self).__init__()

        shape = shape[0] * shape[1] * shape[2]
        small_shape = shape / 2
        self.norm1 = nn.InstanceNorm2d(feat)

        self.linear = nn.Linear(shape,shape)
        self.dwconv = nn.Conv2d(feat, feat,3,stride=1,padding=1, groups=feat)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(shape,shape)
        self.norm2 = nn.InstanceNorm2d(feat)

        self.pool = nn.AvgPool2d(pool_kernel,pool_stride)
        self.linear3 = nn.Linear(small_shape,small_shape)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(small_shape,small_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        res = x
        x = self.norm1(x)

        x = self.linear(x)
        x = self.dwconv(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.norm2(x)

        x2 = self.pool(x)
        x2 = self.linear3(x2)
        x2 = self.relu2(x2)
        x2 = self.linear4(x2)
        x2 = self.sigmoid(x2)

        x = torch.mul(x,x2)
        x = torch.add(x,res)

        return x
        
class ConvBlockNoActive(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size = 3, stride = 1, pad = 1, dilation = 1, groups = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad, dilation, groups)

    def forward(self,x):

        x = self.conv(x)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride,pad, dilation):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_feat, in_feat, kernel_size, stride, pad, dilation, groups=in_feat)
        self.point_conv = nn.Conv2d(in_feat, out_feat, 1)
    
    def forward(self,x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class TransposedUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, kernel = 11, stride = 2, use_csdlkcb = True,pad=60,cuda=hparams["gpu_mode"]):
        super().__init__()
        self.use_csdlkcb = use_csdlkcb
    
        self.csdlkcb = CSDLKCB(in_feat, out_feat,kernel, pad=pad)    # if weird stuff happens disable

        if use_csdlkcb is False:
            padding = 2
        self.up = nn.ConvTranspose2d(out_feat, out_feat, 2, stride, padding = 0)

        if cuda:
            self.csdlkcb = self.csdlkcb.cuda()
            self.up = self.up.cuda()

    def forward(self,x):
        if self.use_csdlkcb is True:
            x = self.csdlkcb(x)

        x = self.up(x)
        #print(out.shape)

        return x

class Upsample(nn.Module):
    def __init__(self, in_feat, scale_factor,cuda=hparams["gpu_mode"]):
        super().__init__()

        self.conv = ConvBlock(in_feat, in_feat * scale_factor * scale_factor, 1, 1,0)
        self.up = nn.PixelShuffle(scale_factor)

        if cuda:
            self.up = self.up.cuda()
            self.conv = self.conv.cuda()
    def forward(self,x):

        x = self.conv(x)
        x = self.up(x)

        return x

class Downsample(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, padding=1, cuda=hparams["gpu_mode"]):
        super(Downsample,self).__init__()

        self.conv = nn.Conv2d(in_feat, out_feat,kernel, stride=2, padding=padding)

        if cuda:
            self.conv = self.conv.cuda()

    def forward(self,x):
        x = self.conv(x)
        return x

class DLKCB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride = 1, pad = 4, group = 4):
        super(DLKCB, self).__init__()
        
        self.pad = nn.ReflectionPad2d((pad, pad, pad, pad))
        self.conv1 = nn.Conv2d(in_feat, in_feat, 1)
        #self.shuffle = ChannelShuffle(groups=group)
        self.dwconv = nn.Conv2d(in_feat, in_feat, kernel,stride, groups=in_feat)
        self.dwdconv = nn.Conv2d(in_feat, in_feat, kernel, stride, dilation=kernel, groups=in_feat)
        self.conv2 = nn.Conv2d(in_feat, out_feat, kernel_size=1)
        
    def forward(self,x):


        x = self.pad(x)

        x = self.conv1(x)

        #x = self.shuffle(x)

        x = self.dwconv(x)

        x = self.dwdconv(x)

        x = self.conv2(x)

        return x

class ResBlock(nn.Module):
    def __init__(self,in_feat, inner_feat, kernel,size, pad,stride=1,groups=4):
        super(ResBlock, self).__init__()

        self.stride = stride
        
        self.pad = nn.ReflectionPad2d((pad, pad, pad, pad))
        self.shuffle = ChannelShuffle(groups=groups)
        self.dwconv = nn.Conv2d(in_feat, in_feat, kernel,1, groups=in_feat)
        self.dwdconv = nn.Conv2d(in_feat, in_feat, kernel, 1, dilation=kernel, groups=in_feat)
        self.norm = nn.LayerNorm((int(size[0]),(int(size[1]))))
        self.conv1 = nn.Conv2d(in_feat, inner_feat, 1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(inner_feat, in_feat, 1,stride=stride)
        
    def forward(self, x):

        res = x
        if self.stride != 1:
            res = F.interpolate(res,scale_factor=0.5)

        x = self.pad(x)
        x = self.shuffle(x)
        x = self.dwconv(x)
        x = self.dwdconv(x)

        x = self.norm(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)

        x = torch.add(x,res)

        return x

class ResBlockFeatChange(nn.Module):
    def __init__(self, in_feat, out_feat, inner_feat, kernel, size, pad, stride=1,groups=4):
        super(ResBlockFeatChange,self).__init__()

        self.res = ResBlock(in_feat,inner_feat,kernel,size,pad,stride,groups=4)
        self.conv = nn.Conv2d(in_feat, out_feat, 1)

    def forward(self,x):
        x = self.res(x)
        x = self.conv(x)
        return x
    
class ResBlockFeatChangeInOut(nn.Module):
    def __init__(self, in_feat, out_feat, inner_feat, kernel, size, pad, stride=1,groups=4,num=0):
        super(ResBlockFeatChangeInOut,self).__init__()

        self.num = num
        self.convin = nn.Conv2d(in_feat,inner_feat,1)
        self.res = ResBlock(inner_feat,inner_feat,kernel,size,pad,stride=1,groups=4)
        self.conv = nn.Conv2d(inner_feat, out_feat, 1, stride=stride)

    def forward(self,x):

        if self.num == 0:
            x = self.convin(x)
            
        x = self.res(x)
        x = self.conv(x)
        return x
    
class MultiResampleOneToMultiple(nn.Module):    # Upsample/Downsample multiple times from single Image (ex. 32) to multiple Img (ex. 128,64,32,16)
    def __init__(self, filter_in, filters_out,use_csdlkcb):
        super(MultiResampleOneToMultiple, self).__init__()

        #filters_out = [128,64,32,16]
        #filter_in = 32
        #use_csdlkcb = True
        filters_out = sorted(filters_out,reverse=True)

        self.filters_out = filters_out

        m = {}
        for filter_out in filters_out:
            if filter_in < filter_out:
                
                down = []
                sort = sorted(filters_out,reverse=True)
                ind_filt_in = sort.index(filter_in)  
                for i in range(ind_filt_in):
                    if i == -1:
                        i = 0


                    down.append(Downsample(sort[ind_filt_in-i],sort[ind_filt_in-i-1]))
                    
                    if sort[ind_filt_in-i-1] == filter_out:
                        break
            
                m[filter_out] = nn.Sequential(*down)

            if filter_in > filter_out:
                
                up = []
                sort = sorted(filters_out)
                ind_filt_in = sort.index(filter_in)  
                for i in range(ind_filt_in):
                    if i == -1:
                        i = 0


                    up.append(TransposedUpsample(sort[ind_filt_in-i],sort[ind_filt_in-i-1]))

                    if sort[ind_filt_in-i-1] == filter_out:
                        break

                m[filter_out] = nn.Sequential(*up)

            if filter_in == filter_out:
                if use_csdlkcb:
                    m[filter_out] = CSDLKCB(filter_in,filter_out)
                else:
                    m[filter_out] = ConvBlock(filter_in,filter_out)

        self.m = m

    def forward(self,x):

        out = {}

        for featnum in self.filters_out:
            out[featnum] = self.m[featnum](x)

        return out
    


class MultiResampleMultipleToOne(nn.Module):    # Upsample/Downsample multiple image to single output shape (ex. 128,64,32,16) to single Img shape (ex. 32)
    def __init__(self, filters_in, filter_out,use_csdlkcb):
        super(MultiResampleMultipleToOne, self).__init__()

        #filters_out = [128,64,32,16]
        #filter_in = 32
        #use_csdlkcb = True
        filters_in = sorted(filters_in,reverse=True)

        self.filters_in = filters_in

        m = {}
        for filter_in in filters_in:
            if filter_in < filter_out:
                
                down = []
                sort = sorted(filters_in,reverse=False)
                ind_filt_in = sort.index(filter_in)  
                for i in range(ind_filt_in):
                    if i == -1:
                        i = 0


                    down.append(Downsample(sort[ind_filt_in-i],sort[ind_filt_in-i-1]))
                    
                    if sort[ind_filt_in-i-1] == filter_out:
                        break
            
                m[filter_in] = nn.Sequential(*down)

            if filter_in > filter_out:
                
                up = []
                sort = sorted(filters_in,reverse=False)
                ind_filt_in = sort.index(filter_in)  
                for i in range(ind_filt_in):
                    if i == -1:
                        i = 0


                    up.append(TransposedUpsample(sort[ind_filt_in-i],sort[ind_filt_in-i-1]))

                    if sort[ind_filt_in-i-1] == filter_out:
                        break

                m[filter_in] = nn.Sequential(*up)

            if filter_in == filter_out:
                if use_csdlkcb:
                    m[filter_in] = CSDLKCB(filter_in,filter_out)
                else:
                    m[filter_in] = ConvBlock(filter_in,filter_out)

        self.m = m

    def forward(self,x):

        out = {}

        for featnum in self.filters_in:
            #print(x[featnum].size())
            out[featnum] = self.m[featnum](x[featnum])
            #print(featnum)
            #print(self.m[featnum])

        return out