#%%
import torch
from train import evaluate, train
from network import Discriminator, Generator, init_weights
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_images = datasets.ImageFolder(root='./Data/resized and sorted', transform=transforms.ToTensor())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D = Discriminator().to(device)
G = Generator().to(device)
### load checkpoints
# D.load_state_dict(torch.load('./Checkpoints/0713/DWeights_500'))
# G.load_state_dict(torch.load('./Checkpoints/0713/GWeights_500'))
D.load_state_dict(torch.load('./Checkpoints/0713/sigmoid/DWeights_400'))
G.load_state_dict(torch.load('./Checkpoints/0713/sigmoid/GWeights_400'))
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
plt.figure(figsize=(9, 3))
for k in range(18):
    plt.subplot(2, 9, k+1)
    plt.gca().set_title('{}'.format(train_images.classes[k]))
    plt.imshow(test_images[k])
plt.show()