import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Estimator():
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train_test_split(self, X, y, test_size=0.2, seed=1334):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test

    def score(self, yhat, y):
        yhat = [1 if i >= 0.5 else 0 for i in yhat]
        auc = roc_auc_score(y, yhat)
        return auc

    def evaluate(self, X, y, n_iters=10):
        scores = [] 
        for iter in range(n_iters):
            X_train, X_test, y_train, y_test = self.train_test_split(X, y, seed=iter)
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            scores.append(self.score(y_pred, y_test))
        return np.array(scores)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )

        self.to(device)

    @staticmethod
    def init_weights(model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
                if layer.bias is not None:
                    init.zeros_(layer.bias)
            
    def fit(self, X, y, epochs=200, lr=0.01, batch_size=32):
        dataset = EmbeddingDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train(loader, epochs=epochs, lr=lr)
        return self.model

    @torch.no_grad()
    def predict(self, X):
        self.model.to('cpu')
        return self.model(torch.from_numpy(X).float()).detach().numpy()

    def forward(self, x):
        return self.model(x)
    
    def train(self, loader, epochs=100, lr=0.01):
        print(f'using {device} device...')
        self.init_weights(self.model)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        pbar = tqdm(range(epochs))
        losses = []

        for epoch in pbar:
            for X, y in loader:
                optimizer.zero_grad()
                y_pred = self.forward(X.to(device))
                loss = self.loss_fn(y_pred, y.to(device))
                loss.backward()
                optimizer.step()
            pbar.set_description(f'Epoch {epoch} loss: {loss}')
            losses.append(loss.item() / len(loader))

        # stop training
        self.model.eval()
        self.losses = losses
    
    def loss_fn(self, y_pred, y):
        return F.mse_loss(y_pred.reshape(-1), y.reshape(-1))
    

class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]