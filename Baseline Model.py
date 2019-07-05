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
#%%
train_images = datasets.ImageFolder(root='C:\\Users\\nucle\\Documents\\GitHub\\PokeGAN\\Data\\resized and sorted', 
                                           transform=transforms.ToTensor())

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x
#%%
def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(train_img_list, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        start_time = time.time()
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        end_time = time.time()
        diff = end_time - start_time
        print(diff)
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs
#%%
model = Autoencoder()
model_underfit = Autoencoder()

max_epochs = 2

outputs = train(model_underfit, num_epochs=max_epochs)
#%%
sample = torch.reshape(train_img_list[0][0], (224,224,3))
plt.imshow(sample.detach().numpy())
#%%

imgs = outputs[max_epochs-1][1].detach().numpy()
plt.subplot(1, 2, 1)
plt.imshow(imgs[0][0], cmap = 'CMRmap_r')
plt.subplot(1, 2, 2)
plt.imshow(imgs[8][0])

#%%
def interpolate(index1, index2):
    x1 = train_images[index1][0]
    x2 = train_images[index2][0]
    x = torch.stack([x1,x2])
    embedding = model_underfit.encoder(x)
    e1 = embedding[0] # embedding of first image
    e2 = embedding[1] # embedding of second image


    embedding_values = []
    for i in range(0, 10):
        e = e1 * (i/10) + e2 * (10-i)/10
        embedding_values.append(e)
    embedding_values = torch.stack(embedding_values)

    recons = model_underfit.decoder(embedding_values)

    plt.figure(figsize=(10, 2))
    for i, recon in enumerate(recons.detach().numpy()):
        plt.subplot(2,10,i+1)
        recon = np.reshape(recon, (224,224,3))
        plt.imshow(recon)
        
    plt.subplot(2,10,11)
    plt.imshow(x2[0])
    plt.subplot(2,10,20)
    plt.imshow(x1[0])
    

interpolate(7, 2)
#%%
interpolate(10,7)

#%%
# Run for less epochs
# Get more data
# Imshow colour

imgs[0][0].shape