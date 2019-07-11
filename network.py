### File where generator and discriminator networks will be defined


#%%
### Import components
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

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
# Test for generator, should return image of noise with 
# input of 64*64 noise
generation = Generator()

testing2 = generation(torch.randn(1, 1, 64, 64))

print(testing2.shape)

img = torch.reshape(testing2, (224,224,3))

plt.imshow(img.detach().numpy())


#%%
### Discriminator Network
