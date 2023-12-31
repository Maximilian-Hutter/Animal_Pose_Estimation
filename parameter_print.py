from models import *
from basic_models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
import time
from models import FeatureTransform

if __name__ == "__main__":
    hparams = {
        "seed": 123,
        "gpus": 1,
        "height":None,
        "width":None,
    }

    np.random.seed(hparams["seed"])    # set seed to default 123 or opt
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])
    gpus_list = range(hparams["gpus"])
    hostname = str(socket.gethostname)
    cudnn.benchmark = True

    # defining shapes

    features = (256,128,64,32)
    feature_out = 32
    Net = FeatureExtract(features,feature_out,num_conv=1,use_csdlkcb=True)


    start = time.time()
    #summary(Net, (32, 256, 256))
    end = time.time()

    proctime = end-start
    print(proctime)

    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    print('===> Building Model ')
    Net = Net


    print('----------------Network architecture----------------')
    print_network(Net)
    print('----------------------------------------------------')