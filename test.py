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
D.load_state_dict(torch.load('./GCloud Checkpoints/128_0720/128_DWeights_900'))
G.load_state_dict(torch.load('./GCloud Checkpoints/128_0720/128_GWeights_900'))
#%%
test_noise = torch.randn(18, 110).to(device)
eye = torch.eye(18)
for i in range(18):  
    ptype_oh = eye[i]
    test_noise[i,:18] = ptype_oh #first 18 contains type info
G.eval()
D.eval()
test_images = G(test_noise).cpu()
test_images = np.moveaxis(test_images.detach().numpy(), 1,3)
plt.figure(figsize=(18, 5))
for k in range(18):
    plt.subplot(2, 9, k+1)
    plt.gca().set_title('{}'.format(train_images.classes[k]))
    plt.imshow(np.clip(test_images[k],0,1))
plt.show()

#%% 36 for each type
def GenNoise(ptype):
    noise = torch.randn(36, 110).to(device)
    eye = torch.eye(18)
    ptype_oh = eye[ptype]
    for i in range(36):
        noise[i,:18] = ptype_oh
    return noise

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

#%% 10 type
# test_noise = torch.randn(10, 110).to(device)
# eye = torch.eye(10)
# for i in range(10):  
#     ptype_oh = eye[i]
#     test_noise[i,:10] = ptype_oh #first 18 contains type info
# G.eval()
# D.eval()
# test_images = G(test_noise).cpu()
# test_images = np.moveaxis(test_images.detach().numpy(), 1,3)
# plt.figure(figsize=(15, 6))
# for k in range(10):
#     plt.subplot(2, 5, k+1)
#     plt.gca().set_title('{}'.format(train_images.classes[k]))
#     plt.imshow(np.clip(test_images[k],0,1))
# plt.show()