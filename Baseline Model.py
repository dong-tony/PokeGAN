#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import random
import torchvision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
train_images = datasets.ImageFolder(root='.\\Data\\resized and sorted', transform=transforms.ToTensor())
#%% data augmentation
trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ToTensor()
])

expanded_dataset = []
for i in range(2):
    train_images_aug = datasets.ImageFolder(root='.\\Data\\resized and sorted', transform=trans)
    for j, item in enumerate(train_images_aug):
        expanded_dataset.append(item)

#%%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.to(device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
#%%
def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3, shuffle = False):
    torch.manual_seed(360)
    rand = random.randint(0,batch_size)
    fig, ax = plt.subplots(1, num_epochs+1, figsize = (num_epochs+1, 1))
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(expanded_dataset, 
                                               batch_size=batch_size, 
                                               shuffle= shuffle)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, label = data[0].to(device), data[1].to(device)
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch:{}/{}, Loss:{:.4f}'.format(epoch+1, num_epochs, float(loss)))
        outputs.append((epoch, img, recon))
        if num_epochs > 1:
            if epoch == 0:
                ax[epoch].imshow(np.moveaxis(outputs[epoch][1][rand].detach().cpu().numpy(),0,2))
                ax[epoch+1].imshow(np.moveaxis(outputs[epoch][2][rand].detach().cpu().numpy(),0,2))
            else:
                ax[epoch+1].imshow(np.moveaxis(outputs[epoch][2][rand].detach().cpu().numpy(),0,2))
        else:
            ax[epoch].imshow(np.moveaxis(outputs[epoch][1][rand].detach().cpu().numpy(),0,2))
            ax[epoch+1].imshow(np.moveaxis(outputs[epoch][2][rand].detach().cpu().numpy(),0,2))
    return outputs
#%%
model = Autoencoder().to(device)
outputs = train(model, num_epochs = 2, shuffle = False)
#%%
def interpolate(model, index1, index2):
    x1 = train_images[index1][0].to(device)
    x2 = train_images[index2][0].to(device)
    x = torch.stack([x1,x2])
    embedding = model.encoder(x)
    e1 = embedding[0] # embedding of first image
    e2 = embedding[1] # embedding of second image

    embedding_values = []
    for i in range(0, 10):
        e = e1 * (i/10) + e2 * (10-i)/10
        embedding_values.append(e)
    embedding_values = torch.stack(embedding_values)

    recons = model.decoder(embedding_values)

    plt.figure(figsize=(10, 2))
    for i, recon in enumerate(recons.detach().cpu().numpy()):
        plt.subplot(2,10,i+1)
        recon = np.moveaxis(recon,0,2)
        plt.imshow(recon)
    
    x1 = np.moveaxis(x1.cpu().numpy(),0,2)
    x2 = np.moveaxis(x2.cpu().numpy(),0,2)
    
    plt.subplot(2,10,11)
    plt.imshow(x2)
    plt.subplot(2,10,20)
    plt.imshow(x1)
#%%
interpolate(model,7,2)