# Animal Pose Estimation for Estimation of Monster Hunter Monster Pose Estimation
# Pose Estimation for Robot creation and 3d Shape Generation
# Pose Estimation for Tail and Wings
# Learning by itself how many Arms/Leg/Wing... strands are necessary i.e. cat needs 5, dog 5, octopus 8 or 9 (8 arms) (9 arms + Head)
# Joint Estimation

import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_models import CSDLKCB, TransposedUpsample, Downsample,ConvBlock, MultiResampleMultipleToOne

class PoseEstimation(nn.Module):    # Final Model
    def __init__(self, features,num_joints,feature_out=1,num_conv=3,use_csdlkcb=True,activation='ReLU',kernelout=3):
        super(PoseEstimation, self).__init__()

        self.num_joints = num_joints

        self.convin = ConvBlock(3,features[0],activation="LeakyReLU")

        jointextracts = []
        self.extract = FeatureExtract(features=features, feature_out=features[0],num_conv=num_conv,use_csdlkcb=use_csdlkcb)
        for i in range(num_joints):
            jointextracts.append(nn.Sequential(JointFeatureExtract(features=features,feature_out=features[0],num_conv=num_conv,use_csdlkcb=use_csdlkcb),ConvBlock(features[0],feature_out,activation=activation,kernel=kernelout,stride=1,pad=1)))
        self.jointextract = jointextracts

    def forward(self,x):

        x = self.convin(x)

        x = self.extract(x)  
        

        out = []
        for i in range(self.num_joints):
            out.append(self.jointextract[i](x))

        return out

class ExtractLadder(nn.Module):
    def __init__(self, filter, num_conv,use_csdlkcb, num_inputs, use_add):
        super(ExtractLadder,self).__init__()

        self.use_add = use_add
        self.num_inputs = num_inputs
        if not use_add:
            filter = filter * num_inputs

        m = []
        if use_csdlkcb:
            for i in range(num_conv):
                m.append(CSDLKCB(filter,filter))
        else:
            for i in range(num_conv):
                m.append(ConvBlock(filter,filter))

        self.model = nn.Sequential(*m)

    def forward(self,x):    # input needs to be tuple or something

        # fuse multiple inputs 
        tens = torch.zeros(x[0].size())

        if self.use_add:
            for i in range(self.num_inputs):
                tens = x[i] +  tens

        else: 
            for i in range(start=0,stop = self.num_inputs, step = 1):
                tens = torch.cat((tens, x[i]), dim = 0)

        #############
        

        out = self.model(tens)

        return out
    
class FeatureTransform(nn.Module):
    def __init__(self, filter_in, filters_out,use_csdlkcb):
        super(FeatureTransform, self).__init__()

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
    

class OutputFuse(nn.Module):
    def __init__(self,features_in,feature_out,use_csdlkcb):
        super().__init__()

        num_in = len(features_in)
        feat_mid = sorted(features_in)[0] * num_in
        self.features_in = sorted(features_in,reverse=True)
        self.resize = MultiResampleMultipleToOne(features_in,feature_out,use_csdlkcb=use_csdlkcb)
        self.conv = ConvBlock(feat_mid,feature_out,"LeakyReLU",1,1,0)
        

    def forward(self,x):

        tens = []
        x = self.resize(x)
        for feat_in in self.features_in:
            #print(feat_in)
            #print(x[feat_in].size())
            tens.append(x[feat_in])

        out = torch.cat(tens,dim=1)

        out = self.conv(out)
            
        return out    

class FeatureExtract(nn.Module):
    def __init__(self,features,feature_out,num_conv,use_csdlkcb):
        super().__init__()

        features = sorted(features)
        self.features = features
        self.full_1 = ExtractLadder(features[0],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)

        self.transform_full_1 = FeatureTransform(features[0],(features[0],features[1]),use_csdlkcb=use_csdlkcb)

        self.full_2 = ExtractLadder(features[0],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.half_2 = ExtractLadder(features[1],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)

        self.transform_full_2 = FeatureTransform(features[0],(features[0],features[1],features[2]),use_csdlkcb=use_csdlkcb)
        self.transform_half_2 = FeatureTransform(features[1],(features[0],features[1],features[2]),use_csdlkcb=use_csdlkcb)

        self.full_3 = ExtractLadder(features[0],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.half_3 = ExtractLadder(features[1],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.quart_3 = ExtractLadder(features[2],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)

        self.transform_full_3 = FeatureTransform(features[0],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)
        self.transform_half_3 = FeatureTransform(features[1],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)
        self.transform_quart_3 = FeatureTransform(features[2],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)

        self.full_4 = ExtractLadder(features[0],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.half_4 = ExtractLadder(features[1],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.quart_4 = ExtractLadder(features[2],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.eight_4 = ExtractLadder(features[3],num_conv=num_conv,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)

        self.transform_full_4 = FeatureTransform(features[0],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)
        self.transform_half_4 = FeatureTransform(features[1],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)
        self.transform_quart_4 = FeatureTransform(features[2],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)
        self.transform_eight_4 = FeatureTransform(features[3],(features[0],features[1],features[2],features[3]),use_csdlkcb=use_csdlkcb)

        self.full_5 = ExtractLadder(features[0],num_conv=0,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.half_5 = ExtractLadder(features[1],num_conv=0,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.quart_5 = ExtractLadder(features[2],num_conv=0,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)
        self.eight_5 = ExtractLadder(features[3],num_conv=0,use_csdlkcb=use_csdlkcb, num_inputs=0, use_add=True)

        self.fuseout = OutputFuse(features_in=features,feature_out=feature_out, use_csdlkcb=use_csdlkcb)      
    def forward(self,x):

        features = sorted(self.features)
        full = self.full_1(x)

        tffull = self.transform_full_1(full)
        full = tffull[self.features[0]]
        half = tffull[self.features[1]]


        full = self.full_2(full)
        half = self.half_2(half)


        tffull = self.transform_full_2(full)
        tfhalf = self.transform_half_2(half)

        full2full = tffull[self.features[0]]
        half2full = tfhalf[self.features[0]]

        full2half = tffull[self.features[1]]
        half2half = tfhalf[self.features[1]]

        full2quart = tffull[self.features[2]]
        half2quart = tfhalf[self.features[2]]


        full = self.full_3((full2full,half2full))
        half = self.half_3((full2half,half2half))
        quart = self.quart_3((full2quart,half2quart))


        tffull = self.transform_full_3(full)
        tfhalf = self.transform_half_3(half)
        tfquart = self.transform_quart_3(quart)

        full2full = tffull[self.features[0]]
        half2full = tfhalf[self.features[0]]
        quart2full = tfquart[self.features[0]]

        full2half = tffull[self.features[1]]
        half2half = tfhalf[self.features[1]]
        quart2half = tfquart[self.features[1]]

        full2quart = tffull[self.features[2]]
        half2quart = tfhalf[self.features[2]]
        quart2quart = tfquart[self.features[2]]

        full2eight = tffull[self.features[3]]
        half2eight = tfhalf[self.features[3]]
        quart2eight = tfquart[self.features[3]]


        full = self.full_4((full2full,half2full,quart2full))
        half = self.half_4((full2half,half2half,quart2half))
        quart = self.quart_4((full2quart,half2quart,quart2quart))
        eight = self.eight_4((full2eight,half2eight,quart2eight))

        tffull = self.transform_full_4(full)
        tfhalf = self.transform_half_4(half)
        tfquart = self.transform_quart_4(quart)
        tfeight = self.transform_eight_4(eight)


        full2full = tffull[self.features[0]]
        half2full = tfhalf[self.features[0]]
        quart2full = tfquart[self.features[0]]
        eight2full = tfeight[self.features[0]]

        full2half = tffull[self.features[1]]
        half2half = tfhalf[self.features[1]]
        quart2half = tfquart[self.features[1]]
        eight2half = tfeight[self.features[1]]

        full2quart = tffull[self.features[2]]
        half2quart = tfhalf[self.features[2]]
        quart2quart = tfquart[self.features[2]]
        eight2quart = tfeight[self.features[2]]

        full2eight = tffull[self.features[3]]
        half2eight = tfhalf[self.features[3]]
        quart2eight = tfquart[self.features[3]]
        eight2eight = tfeight[self.features[3]]


        full = self.full_5((full2full,half2full,quart2full,eight2full))
        half = self.half_5((full2half,half2half,quart2half,eight2half))
        quart = self.quart_5((full2quart,half2quart,quart2quart,eight2quart))
        eight = self.eight_5((full2eight,half2eight,quart2eight,eight2eight))


        x = (full,half,quart,eight)
        tensdict = {}

        for i,featnum in enumerate(features):
            tensdict[featnum] = x[i]
            #print(featnum)
            #print(x[i].size())

        out = self.fuseout(tensdict)



        return out

class JointFeatureExtract(nn.Module):
    def __init__(self,features,feature_out=1,num_conv=3,use_csdlkcb = True):
        super(JointFeatureExtract, self).__init__()

        self.extract = FeatureExtract(features=features,feature_out=feature_out,num_conv=num_conv,use_csdlkcb=use_csdlkcb)

    def forward(self,x):

        out = self.extract(x)
        return out