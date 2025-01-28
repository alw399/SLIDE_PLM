import numpy as np 
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multioutput import MultiOutputRegressor


class EmbeddingEvaluator():
    def __init__(self, model=None):

        self.model = model
        if model is None:
            self.model = MultiOutputRegressor(MLPRegressor())


    @staticmethod
    def zero_out_dim(embed, dim):
        '''
        Zero out a dimension of the embedding
        @param embedding: the embedding to zero out
        @param dim: the dimension to zero out
        @return embedding: the embedding with the specified dimension zeroed out
        '''
        embedding = embed.copy()
        embedding[:, dim] = 0
        return embedding
    
    def predict(self, X, y):
        clf = self.model
        clf.fit(X, y)
        yhat = clf.predict(X)

        loss = yhat - y
        return loss

    def get_contributions(self, embedding, y):

        baseline = self.predict(embedding, y)

        losses = [baseline]
        for i in tqdm(range(embedding.shape[1]), desc='Performing deletion experiments...'):
            zeroed_embedding = self.zero_out_dim(embedding, i)
            loss = self.predict(zeroed_embedding, y)
            losses.append(loss)

        return np.array(losses)






