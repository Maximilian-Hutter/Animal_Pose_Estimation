import torchvision.transforms as T
import os
from PIL import Image
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import time
import torch
import cv2
import numpy as np
from math import *
import random

def random_choose_images(inpath, outpath):
    files = os.listdir(inpath)

    for i,file in enumerate(files):
        if random.randint(0,30) % 30 == 0:
            image = Image.open(inpath + file)
            image.save(outpath + file + ".png")


def pil_to_opencv(image):
    open_cv_image = np.array(image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def psnr(original, processed):
    original = pil_to_opencv(original)
    processed = pil_to_opencv(processed)
    mse = np.mean((original - processed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def sizediv(size, div):

    size0 = size[0] / div
    size1 = size[1] / div

    out = (int(size0),int(size1))

    return out

def print_size(Net):
    param_size = 0
    for param in Net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in Net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def calc_accuracy(gen, label, size):

    sizeout = 1
    for num in size:
        sizeout = num + sizeout
    
    train_acc = torch.isclose(gen,label)
    train_acc = train_acc.sum()
    train_acc = torch.div(train_acc, sizeout)

def extractFrames(pathIn, pathOut):
    #os.mkdir(pathOut)
    cap = cv2.VideoCapture(pathIn)
    count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def checkpointGenerate(epoch, hparams, Net,optimizer,loss, name = None):
    model_out_path = hparams["save_folder"]+str(epoch)+ hparams["model_type"]+".pth".format(epoch)
    torch.save({
            'epoch': epoch,
            'model_state_dict': Net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_out_path)
            
    print("Checkpoint saved to {}".format(model_out_path))


def print_network(net, hparams):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    print('===> Building Model ', hparams["model_type"])
    if hparams["model_type"] == 'VQGAN':
        Net = Net



def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def save_trainimg(generated_image, epoch, name = ""):
    gimg = torch.split(generated_image,1)
    gimg = gimg[0]
    transform = T.ToPILImage()
    #norm = T.Normalize(mean=[-1,-1,-1],std=[2,2,2])
    #generated_image = norm(generated_image)

    gimg = transform(gimg.squeeze(0))


    if not os.path.isdir("trainimg"):
        os.mkdir("trainimg")
    gimg.save("trainimg/"+ str(name) + str(epoch) +".png")

def save_tensorimg(image, name = ""):
    image = torch.split(image.clone(),1, dim = 1)
    transform = T.ToPILImage()
    image = image[0].squeeze(0)
    image = transform(image)
    if not os.path.isdir("debug_images"):
        os.mkdir("debug_images")

    image.save("debug_images/" + str(name)+".png")

def save_allimg(generated_image, label, input, epoch):
    transform = T.ToPILImage()
    gimg = transform(generated_image.squeeze(0))
    lab = transform(label.squeeze(0))
    inp = transform(input.squeeze(0))
    if not os.path.isdir("trainimg"):
        os.mkdir("trainimg")
    gimg.save("trainimg/gen"+ str(epoch) +".png")
    lab.save("trainimg/label"+str(epoch) + ".png")
    inp.save("trainimg/input"+ str(epoch) +".png")


def prepare_imgdatadir(path, outpath, substring = None, numerate = False, startnum = 0, size= None, crop_size = None, multiple_dirs = False, five_crop=False, first_section_only=False):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    all_files = []
    if multiple_dirs is True:
        dirs = os.listdir(path)
        for dir in dirs:
            files = os.listdir(path + "/" + dir)
            outfiles = []
            if substring != None:
                for file in files:
                    if substring in file:
                        outfiles.append(file)
            for file in outfiles:
                all_files.append(dir + "/" + file)
    else:
        files = os.listdir(path)
        outfiles = []
        if substring != None:
            for file in files:
                if substring in file:
                    outfiles.append(file)
        else:
            outfiles = files

        all_files = outfiles


    for i,file in enumerate(BackgroundGenerator(tqdm(all_files), max_prefetch=5), start=startnum):
            img = Image.open(path + "/" + file)


            if five_crop == True:
                crop = T.FiveCrop(size=size)
                imgtup = crop(img)
                for a,img in enumerate(imgtup):
                    img.save(outpath +str(i) +"_" +str(a) + ".png")


            elif size != None:
                img = img.resize(size)
            if crop_size != None:
                img = crop_center(img, crop_size[0], crop_size[1])
            if first_section_only is True:
                file = file.split("_")
                img.save(outpath + file[0] + ".png")
            if numerate:
                img.save(outpath + str(i) + ".png")
           # else:
                #img.save(outpath + file)


if __name__ == "__main__":
    random_choose_images("./results/", "./chosen_pics/")
    #prepare_imgdatadir("C:/Data/dehaze/SOTS/outdoor/input/", "C:/Data/dehaze/SOTS/outdoor/prepped/" ,substring = None ,numerate=False,startnum = 1, size=(640,360), multiple_dirs=False, five_crop=False, first_section_only=True)