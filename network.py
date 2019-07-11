#%%
### Import components
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms

#%%
### Generator networks
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(64*64, 32, 7),
            nn.BatchNorm2d(32, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=4, padding=1, output_padding=1),
            nn.BatchNorm2d(16, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 8, 3, stride=3, padding=2, output_padding=1),
            nn.BatchNorm2d(8, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(8, 4, 3, stride=3, padding=1, output_padding=1),
            nn.BatchNorm2d(4, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(4, 3, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(x.size(0), 64*64, 1, 1)
        out = self.model(x)
        return out
#%%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            # Leaving layer for future use
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(16, 32, 7)
        )
    
        # discriminator fc
        self.fc_dis = nn.Linear(56*56*16, 1)
        # aux-classifier fc, 18 different types
        self.fc_aux = nn.Linear(56*56*16, 18)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        flattened = x.view(-1, 56*56*16)
        fc_dis = self.fc_dis(flattened)
        authenticity = self.sigmoid(fc_dis)
        fc_aux = self.fc_aux(flattened)
        ptype = self.softmax(fc_aux)
        return authenticity, ptype

# #%%
# # Test for generator, should return image of noise with 
# # input of 64*64 noise
# generation = Generator()

# testing2 = generation(torch.randn(1, 1, 64, 64))

# print(testing2.shape)

# img = torch.reshape(testing2, (224,224,3))

# plt.imshow(img.detach().numpy())
# #%%
# train_images = datasets.ImageFolder(root='.\\Data\\resized and sorted', 
#                                            transform=transforms.ToTensor())
# # Test for discriminator model, should retrun 2 tensors
# model = Discriminator()
# testing = model(train_images[0][0].unsqueeze(0))

# testing