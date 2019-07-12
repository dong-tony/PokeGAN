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
def train(generator, discriminator, batch_size = 1, d_lr=0.0002, g_lr=0.0002, num_epochs=5, save = False, name = 'Model x'):
    dis_criterion = nn.BCEWithLogitsLoss()
    aux_criterion = nn.CrossEntropyLoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas= (0.5,0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas= (0.5,0.999))

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
            dis_label = torch.zeros(pokemon.shape[0], 1).to(device)
            aux_label = ptype
            dis_output, aux_output = discriminator(pokemon)
            dis_loss_real = dis_criterion(dis_output, dis_label)
            aux_loss_real = aux_criterion(aux_output, aux_label)
            DLoss_real = dis_loss_real + aux_loss_real
            DLoss_real.backward()
            D_x = torch.sigmoid(dis_output.data.mean())
            aux_acc = evaluate(aux_output, aux_label)
            
            # training with fake images
            noise = torch.randn(pokemon.shape[0], 110).to(device)
            ptype_fake = torch.randint(0, 18, (pokemon.shape[0],), dtype = torch.long).to(device)
            eye = torch.eye(18)
            ptype_oh = eye[ptype_fake]
            noise[:,:18] = ptype_oh #first 18 contains type info
            pokemon_fake = generator(noise)
            dis_label = torch.ones(pokemon.shape[0], 1).to(device)
            aux_label = ptype_fake
            dis_output, aux_output = discriminator(pokemon_fake)
            dis_loss_fake = dis_criterion(dis_output, dis_label)
            aux_loss_fake = aux_criterion(aux_output, aux_label)
            DLoss_fake = dis_loss_fake + aux_loss_fake
            DLoss_fake.backward()
            D_G_z = torch.sigmoid(dis_output.data.mean())

            DLoss = DLoss_real + DLoss_fake
            d_optimizer.step()

            # === Train the Generator ===
            g_optimizer.zero_grad()
            pokemon_fake = generator(noise)        
            dis_label = torch.zeros(pokemon.shape[0], 1).to(device) #real
            aux_label = ptype_fake
            dis_output, aux_output = discriminator(pokemon_fake)
            dis_loss_G = dis_criterion(dis_output, dis_label)
            aux_loss_G = aux_criterion(aux_output, aux_label)
            GLoss = dis_loss_G + aux_loss_G
            GLoss.backward()
            g_optimizer.step()

        print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.4f, D(G(z)): %.4f, Class_acc: %.4f' 
            % (epoch + 1, num_epochs, DLoss.item(), GLoss.item(), D_x, D_G_z, aux_acc))
        
        # plot images
        test_noise = torch.randn(18, 110).to(device)
        eye = torch.eye(18)
        for i in range(18):  
            ptype_oh = eye[i]
            test_noise[i,:18] = ptype_oh #first 18 contains type info
        generator.eval()
        discriminator.eval()
        test_images = generator(test_noise).cpu()
        test_images = np.moveaxis(test_images.detach().numpy(), 1,3)
        plt.figure(figsize=(9, 3))
        for k in range(18):
            plt.subplot(2, 9, k+1)
            plt.gca().set_title('{}'.format(train_images.classes[k]))
            plt.imshow((test_images[k]+1)/2)
        plt.show()
        
        if save and (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), '{}_GWeights_{}'.format(name, (epoch + 1)))
            torch.save(discriminator.state_dict(), '{}_DWeights_{}'.format(name, (epoch + 1)))

def evaluate(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data))
    return acc