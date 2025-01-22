import numpy as np 
import pandas as pd
from sklearn.neural_network import MLPClassifier


class EmbeddingEvaluator():
    def __init__(self):
        pass

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
    
    @staticmethod
    def predict(X, y):
        clf = MLPClassifier()
        clf.fit(X, y)

        accuracy = clf.score(X, y)
        return accuracy

    def get_contributions(self, embedding, y):

        baseline = self.predict(embedding, y)

        accuracies = [baseline]
        for i in range(embedding.shape[1]):
            zeroed_embedding = self.zero_out_dim(embedding, i)
            accuracy = self.predict(zeroed_embedding, y)
            accuracies.append(accuracy)

        return accuracies






