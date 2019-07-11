#%%
import torch
import torch.nn as nn

#%%
train_images = datasets.ImageFolder(root='C:\\Users\\nucle\\Documents\\GitHub\\PokeGAN\\Data\\resized and sorted', 
                                           transform=transforms.ToTensor())

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
        realfake = self.sigmoid(fc_dis)
        fc_aux = self.fc_aux(flattened)
        classes = self.softmax(fc_aux)
        return realfake, classes

#%%
# Test for discriminator model, should retrun 2 tensors
model = Discriminator()
testing = model(train_images[0][0].unsqueeze(0))

testing