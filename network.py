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
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 256, 3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 192, 5, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, 5, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
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

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)