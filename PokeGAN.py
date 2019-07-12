#%%
import torch
from train import evaluate, train
from network import Discriminator, Generator, init_weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
D = Discriminator().to(device)
G = Generator().to(device)
D.apply(init_weights)
G.apply(init_weights)
### load checkpoints
D.load_state_dict(torch.load('./Checkpoints/0712 Collab/DWeights650'))
G.load_state_dict(torch.load('./Checkpoints/0712 Collab/DWeights650'))
train(G, D, batch_size = 50, num_epochs = 300, save = True, name = '0712 Model 1')