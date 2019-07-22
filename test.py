#%%
import torch
from train import evaluate, train
# from network import Discriminator, Generator, init_weights
# from EasyColor.network_EasyColor import Discriminator, Generator
from Small128.network_128 import Discriminator, Generator, init_weights
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision import datasets, transforms
import numpy as np
import os, glob
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_images = datasets.ImageFolder(root='./Data/resized and sorted', transform=transforms.ToTensor())
#train_images = datasets.ImageFolder(root='./Data/resized and sorted_EasyColor', transform=transforms.ToTensor())
D = Discriminator().to(device)
G = Generator().to(device)
### load checkpoints
# D.load_state_dict(torch.load('./GCloud Checkpoints/aux_loss2_aug/0718_DWeights_500'))
# G.load_state_dict(torch.load('./GCloud Checkpoints/aux_loss2_aug/0718_GWeights_500'))
# D.load_state_dict(torch.load('./GCloud Checkpoints/EasyColor_0720/EasyColor_DWeights_450'))
# G.load_state_dict(torch.load('./GCloud Checkpoints/EasyColor_0720/EasyColor_GWeights_450'))
# D.load_state_dict(torch.load('./Checkpoints/0714/DWeights_600'))
# G.load_state_dict(torch.load('./Checkpoints/0714/GWeights_600'))
D.load_state_dict(torch.load('./GCloud Checkpoints/128_0720/128_DWeights_1100', map_location= 'cpu'))
G.load_state_dict(torch.load('./GCloud Checkpoints/128_0720/128_GWeights_1100', map_location= 'cpu'))
#%% 36 for each type
def GenNoise(ptype):
    noise = torch.randn(36, 110).to(device)
    eye = torch.eye(18)
    ptype_oh = eye[ptype]
    for i in range(36):
        noise[i,:18] = ptype_oh
    return noise

def GenPokemon():
    G.eval()
    D.eval()
    for ptype in range(18):
        images = G(GenNoise(ptype)).cpu()
        images = np.moveaxis(images.detach().numpy(), 1,3)
        plt.figure(figsize=(10,10))
        for i in range(36):
            plt.subplot(6, 6, i+1)
            plt.gca().axis('off')
            plt.imshow(np.clip(images[i],0,1))
            plt.suptitle('{}'.format(train_images.classes[ptype]))
        plt.show()

GenPokemon()
#%% Color histogram comparison
import cv2
from PIL import Image
def GenNoise(ptype):
    noise = torch.randn(50, 110).to(device)
    eye = torch.eye(18)
    ptype_oh = eye[ptype]
    for i in range(50):
        noise[i,:18] = ptype_oh
    return noise
    
def GenPokeArray():
    G.eval()
    D.eval()
    PokeArray = []
    for ptype in range(18):
        images = G(GenNoise(ptype)).cpu()
        images = np.moveaxis(images.detach().numpy(), 1,3)
        PokeArray.append(images*255)
    return PokeArray

def AvgImage(path, size, file = True):
    avg = np.zeros((size,size,3))
    if file == True: 
        for i, image in enumerate(glob.iglob("{}/*.png".format(path))):
            image_array = np.asarray(Image.open(image))[:,:,0:3]
            avg += image_array
        avg = avg/i
    else:
        for i in range(50):
            avg += path[i]
        avg = avg/i
    return avg

def GenAvgs(folder, size, file = True):
    avgs = []
    if file == True:
        for ptype in train_images.classes:
            avgs.append(AvgImage('{}/{}'.format(folder,ptype), size))
    else:
        for array in folder:
            avgs.append(AvgImage(array, size, file = False))
    return avgs
real = GenAvgs('Data/resized and sorted_128/', 128)
gen = GenAvgs(GenPokeArray(), 128, file = False)
# GenHistogram(real, gen)
#%%
def GenHistogramComp(real_avgs, gen_avgs):
    for i in range(18):
        avg_real = real_avgs[i].astype(np.uint8)
        avg_gen = gen_avgs[i].astype(np.uint8)
        color = ('r','g','b')
        for j, col in enumerate(color):
            hist_real = cv2.calcHist([avg_real],[j],None,[256],[0.1,256])
            hist_gen = cv2.calcHist([avg_gen],[j],None,[256],[0.1,256])
            plt.plot(hist_real, '-', color = col, alpha = 0.3)
            plt.plot(hist_gen,':', color = col)
            plt.xlim([0,256])
            plt.title('{}'.format(train_images.classes[i]))
        plt.show()
def GenHistogram(avgs):
    for i in range(18):
        avg = avgs[i].astype(np.uint8)
        color = ('r','g','b')
        for j, col in enumerate(color):
            hist = cv2.calcHist([avg],[j],None,[256],[0,256])
            plt.plot(hist, color = col)
            plt.xlim([0,256])
            plt.title('{}'.format(train_images.classes[i]))
        plt.show()
#GenHistogram(gen)
GenHistogramComp(real, gen)