from dataset.ap10k import AnimalAP10KDataset
from models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from config import _C
# my imports
import myutils
#from loss import ChabonnierLoss, ColorLoss, GrayscaleLoss
from params import hparams
from training import Train

if __name__ == '__main__':

    np.random.seed(hparams["seed"])    # set seed to default 123 or opt
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])
    gpus_list = range(hparams["gpus"])
    cuda = hparams["gpu_mode"]
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(hparams)

    cfg = _C.clone()
    #cfg.merge_from_file("cfg.yaml")
    cfg.freeze()
    #size = (hparams["height"], hparams["width"])

    print('==> Loading Datasets')
    dataloader = DataLoader(AnimalAP10KDataset(cfg, hparams["train_data_path"], 'train', 1, None))

    # define the Network
    Net = PoseEstimation(features=hparams["features"],num_joints=hparams["num_joints"],feature_out=hparams["feature_out"],num_conv=hparams["num_conv"],use_csdlkcb=hparams["use_csdlkcb"],activation=hparams["activation"],kernelout=hparams["kernelout"])
    # print Network parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # set criterions & optimizers
    criterion = nn.L1Loss()
    optimizer = optim.Adam(Net.parameters(), lr=hparams["lr"], betas=(hparams["beta1"],hparams["beta2"]))

    cuda = hparams["gpu_mode"]
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    torch.manual_seed(hparams["seed"])
    if cuda:
        torch.cuda.manual_seed(hparams["seed"])

    if cuda:
        Net = Net.to(torch.device("cuda:0"))
        criterion = criterion.to(torch.device("cuda:0"))
        #color_crit = color_crit.to(torch.device("cuda:0"))

    # load checkpoint/load model
    star_n_iter = 0
    start_epoch = 0
    if hparams["resume"]:
        checkpoint = torch.load(hparams["resume_train"]) ## look at what to load
        #start_epoch = checkpoint['epoch']
        #start_n_iter = checkpoint['n_iter']
        #optimizer.load_state_dict(checkpoint['optim'])
        print("last checkpoint restored")


    param_size = 0
    for param in Net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in Net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    training = Train(hparams, Net, optimizer, criterion)

    training.train(dataloader, "ap10k", hparams["epochs"])

    myutils.print_network(Net, hparams)