#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
train_images = datasets.ImageFolder(root='.\\Data\\resized and sorted', transform=transforms.ToTensor())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
def train(generator, discriminator, batch_size = 1, lr=0.001, num_epochs=5):
    dis_criterion = nn.BCEWithLogitsLoss()
    aux_criterion = nn.CrossEntropyLoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):

        generator.train()
        discriminator.train()

        for n, (pokemon, ptype) in enumerate(train_loader):
            ####################################
            # === Train the Discriminator   ===
            ####################################
            # training with real images
            d_optimizer.zero_grad()
            pokemon, ptype = pokemon.to(device), ptype.to(device)
            dis_label = torch.zeros(batch_size)
            aux_label = ptype
            dis_output, aux_output = discriminator(pokemon)
            dis_loss_real = dis_criterion(dis_output, dis_label)
            aux_loss_real = aux_criterion(aux_output, aux_label)
            DLoss_real = dis_loss_real + aux_loss_real
            DLoss_real.backward()
            
            # training with fake images
            noise = torch.randn(batch_size, 1, 64*64)
            ptype_fake = np.random(0, 18, (batch_size,), dtype = torch.long)
            eye = torch.eye((batch_size, 18))
            ptype_oh = eye[ptype_fake]
            noise[:,:,:18] = ptype_oh #first 18 contains type info
            pokemon_fake = generator(noise)
            dis_label = torch.ones(batch_size)
            aux_label = ptype_fake
            dis_output, aux_output = discriminator(pokemon_fake)
            dis_loss_fake = dis_criterion(dis_output, dis_label)
            aux_loss_fake = aux_criterion(aux_output, aux_label)
            DLoss_fake = dis_loss_fake + aux_loss_fake
            DLoss_fake.backward()

            DLoss = DLoss_real + DLoss_fake
            d_optimizer.step()

            # === Train the Generator ===
            g_optimizer.zero_grad()
            dis_label = torch.zeros(batch_size) #real
            aux_label = ptype_fake
            dis_output, aux_output = discriminator(pokemon_fake)
            dis_loss_G = dis_criterion(dis_output, dis_label)
            aux_loss_G = aux_criterion(aux_output, aux_label)
            GLoss = dis_loss_G + aux_loss_G
            GLoss.backward()
            g_optimizer.step()

        # scores = torch.sigmoid(d_outputs)
        # real_score = scores[:images.size(0)].data.mean()
        # fake_score = scores[images.size(0):].data.mean()


        # print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
        #       'D(x): %.2f, D(G(z)): %.2f' 
        #       % (epoch + 1, num_epochs, d_loss.item(), g_loss.item(), real_score, fake_score))
        
        # plot images
        test_noise = torch.randn(16, 1, 64*64)
        generator.eval()
        discriminator.eval()
        test_images = generator(test_noise)
        plt.figure(figsize=(9, 3))
        for k in range(16):
            plt.subplot(2, 8, k+1)
            plt.imshow(test_images[k,:])
        plt.show()