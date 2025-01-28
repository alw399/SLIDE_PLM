import torch 
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
import torch.optim as optim

from tqdm import tqdm


class SimpleNN():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )

    @staticmethod
    def init_weights(model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
                if layer.bias is not None:
                    init.zeros_(layer.bias)
            
    def fit(self, X, y, epochs=200, lr=0.01):
        dataset = EmbeddingDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        self.train(loader, epochs=epochs, lr=lr)
        return self.model

    def predict(self, X):
        return self.model(torch.from_numpy(X).float()).detach().numpy()

    def forward(self, x):
        return self.model(x)
    
    def train(self, loader, epochs=100, lr=0.01):
        self.init_weights(self.model)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        pbar = tqdm(range(epochs))
        losses = []

        for epoch in pbar:
            for X, y in loader:
                optimizer.zero_grad()
                y_pred = self.forward(X)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
            pbar.set_description(f'Epoch {epoch} loss: {loss}')
            losses.append(loss.item() / len(loader))

        # stop training
        self.model.eval()
        self.losses = losses
    
    def loss_fn(self, y_pred, y):
        return nn.MSELoss()(y_pred, y)
    

class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]