from models import PoseEstimation
import torch
import myutils
tens = torch.rand((1,3,512,512)).cpu()

fe = PoseEstimation(features=(8,16,32,64),num_joints=18,feature_out=1,num_conv=3,use_csdlkcb=True,activation="LeakyReLU",kernelout=3)
tensout = fe(tens)

print(tensout.__len__())
for i,tens in enumerate(tensout):
    myutils.save_tensorimg(tens, "jointmap num" + str(i))
