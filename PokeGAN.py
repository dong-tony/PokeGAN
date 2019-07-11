#%%
import torch
from train import train
from network import Discriminator, Generator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
D = Discriminator().to(device)
G = Generator().to(device)
train(G, D, batch_size = 10, lr = 0.001, num_epochs = 1)