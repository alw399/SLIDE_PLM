import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn as nnyy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import torch.optim as optim
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser('Test VAE to downsize embeddings (HL2)')
parser.add_argument('--plm_vectors',required=True,help='pLM vectors (pickle)')
parser.add_argument('--input_dim',required=True,help='input dim',default=128)
parser.add_argument('--hidden1_dim',required=True,help='hidden1 dim',default=64)
parser.add_argument('--latent_dim',required=True,help='latent dim',default=2)
parser.add_argument('--hidden2_dim',required=True,help='hidden2 dim (reduced size)',default=4)
parser.add_arguemnt('--output_vec_name',required=True,help='name for reduced vectors')

# already embedded previously!
with open(args.plm_vectors, 'rb') as f:  
    vecs = pickle.load(f) # serialize the list
print('Number of Vectors: '+str(len(vecs)))

# Define the VAE model
class VAE1D(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim1, hidden_dim2): # 128, 2, 64?, 4 (SLIDE)
        super(VAE1D, self).__init__()
        
        # Encoder (reducing the input to hidden1 size)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Input layer: reduce inpt -> hidden1
        
        # latent mean and var
        self.fc2_mean = nn.Linear(hidden_dim1, latent_dim)  # Mean of latent space: map hidden1 -> latent_dim
        self.fc2_log_var = nn.Linear(hidden_dim1, latent_dim)  # Log variance: map hidden1 -> latent_dim

        # Decoder (reconstructing from latent space to hidden2 (SLIDE size) and back to input size)
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)  # maps latent space to hidden2: latent_dim -> hidden2
        self.fc4 = nn.Linear(hidden_dim2, input_dim)  # maps hidden2 to input: hidden2 -> input
   
    def encode(self, x):
        h1 = torch.relu(self.fc1(x)) # relu activation
        mean = self.fc2_mean(h1)
        log_var = self.fc2_log_var(h1)
        return mean, log_var 

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h2 = torch.relu(self.fc3(z))
        return self.fc4(h2), h2

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x, hidden_layer2 = self.decode(z)
        return recon_x, mean, log_var, hidden_layer2 # return hidden layer 2

# Loss function for VAE - KL divergece + reconstruction loss
def vae_loss(recon_x, x, mean, log_var):
    recon_loss = nn.MSELoss()(recon_x, x) # cross entropy loss?
    kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kld_loss

# 1D data
tensor_data = torch.from_numpy(np.array(vecs))
dataloader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)

# Model, optimizer, and training
input_dim = args.input_dim # pLM size
latent_dim = args.latent_dim
hidden_dim1 = args.hidden1_dim # larger or smaller?
hidden_dim2 = args.hidden2_dim  # SLIDE
vae = VAE1D(input_dim, latent_dim, hidden_dim1, hidden_dim2) # 128->64->2->4->128
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
epochs = 50
loss_plot = []
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        batch = batch[0]  # Extract data from batch
        optimizer.zero_grad()
        recon_batch, mean, log_var, hidden_layer2 = vae(batch)
        loss = vae_loss(recon_batch, batch, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_plot.append(total_loss/len(dataloader))
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
print("Training complete.")

# Testing the VAE and saving hidden layer
vae.eval()  # Set model to evaluation mode
with torch.no_grad():
    test_sample = tensor_data # everything ????
    reconstructed_samples, _, _, hidden_layer2 = vae(test_sample)

    # Save hidden layers to a file
    hidden_layer2_np = hidden_layer2.numpy()
    np.save(args.output_name+"_ReducedVecs.npy", hidden_layer2_np)  # Save as .npy file

    # Evaluate reconstruction metrics
    test_sample_np = test_sample.numpy()
    reconstructed_samples_np = reconstructed_samples.numpy()

    mse = mean_squared_error(test_sample_np, reconstructed_samples_np)
    mae = mean_absolute_error(test_sample_np, reconstructed_samples_np)

    print("Mean Squared Error (MSE) on test data:", mse)
    print("Mean Absolute Error (MAE) on test data:", mae)

# plot loss
plt.plot(loss_plot,range(epochs))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(args.output_name+'_LossPlot.png')
