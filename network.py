#%%
### Import components
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#%%
### Generator networks
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(110*1*1, 768)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 3, stride = 2, padding = 0),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(384, 256, 3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 192, 5, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(192, 64, 5, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 8, stride=2, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 110)
        x = self.fc(x)
        x = x.view(-1, 768, 1, 1)
        out = self.model(x)
        return out
#%%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),    
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),     
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),  
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)   
        )
        # discriminator fc
        self.fc_dis = nn.Linear(512*14*14, 1)
        # aux-classifier fc, 18 different types
        self.fc_aux = nn.Linear(512*14*14, 18)

    def forward(self, x):
        x = self.model(x)
        flattened = x.view(-1, 512*14*14)
        fc_dis = self.fc_dis(flattened)
        authenticity = fc_dis
        fc_aux = self.fc_aux(flattened)
        ptype = fc_aux
        return authenticity, ptype

# #%%
# # Test for generator, should return image of noise with 
# # input of 64*64 noise
# generation = Generator()

# testing2 = generation(torch.randn(1, 1, 64, 64))

# print(testing2.shape)

# img = torch.reshape(testing2, (224,224,3))

# plt.imshow(img.detach().numpy())
#%%
train_images = datasets.ImageFolder(root='.\\Data\\resized and sorted', 
                                           transform=transforms.ToTensor())
# Test for discriminator model, should retrun 2 tensors
train_loader = torch.utils.data.DataLoader(train_images, batch_size=50, shuffle=True)
for pokemon, ptype in train_loader:
    model = Discriminator()
    testing = model(pokemon)[0]
    print(testing.shape)
    break