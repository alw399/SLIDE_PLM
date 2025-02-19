import numpy as np 
import pandas as pd 
import os, pickle

from sklearn.linear_model import LinearRegression, Lasso
from knockpy import KnockoffFilter

from util import get_sigLFs

class Interaction():

    def __init__(self, slide_outs, plm_embed=None, y=None, interacts_only=False, model='lasso'):
        self.slide_outs = slide_outs
        self.sig_LFs = get_sigLFs(slide_outs)
        self.interacts_only = interacts_only

        self.plm_embedding = plm_embed
        self.y = y

        self.z_matrix = self.get_z_matrix(interacts_only=interacts_only)
        self.n, self.k = self.z_matrix.shape
        self.l = self.plm_embedding.shape[1] 

        self.interaction_terms = self.get_interaction_terms(self.z_matrix, self.plm_embedding)

        if model == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif model == 'LR':
            self.model = LinearRegression()
        else:
            raise ValueError('Model not supported')

    def get_z_matrix(self, interacts_only=True):
        z_matrix = pd.read_csv(os.path.join(self.slide_outs, 'z_matrix.csv'), index_col=0)
        z_matrix = z_matrix[self.sig_LFs].values

        if not interacts_only:
            z_matrix = np.hstack([z_matrix, np.ones((z_matrix.shape[0], 1))])

        return z_matrix

    @staticmethod
    def get_interaction_terms(z_matrix, plm_embedding):
        '''
        @return: interactions in shape of (n_samples, n_LFs, plm_embed_dim)
        '''
        return np.einsum('ij,ik->ijk', z_matrix, plm_embedding)

    @staticmethod
    def filter_knockoffs(interaction_terms, y, fdr=0.05):
        '''
        @return: mask of 0,1 significant interaction terms
        '''
        # # This uses gaussian maxent knockoffs
        # kfilter = KnockoffFilter(ksampler='gaussian', knockoff_kwargs={'method':'maxent'})

        # # This uses fixed-X SDP knockoffs
        # kfilter = KnockoffFilter(ksampler='fx', knockoff_kwargs={'method':'sdp'})

        # # Metropolized sampler for heavy-tailed t markov chain using MVR-guided proposals
        # kfilter = KnockoffFilter(ksampler='artk', knockoff_kwargs={'method':'mvr'})

        kfilter = KnockoffFilter(
            ksampler='gaussian', 
            fstat='lasso'
        )

        rejections = kfilter.forward(X=interaction_terms, y=y, fdr=fdr, shrinkage="ledoitwolf")
        return rejections
    
    def fit_linear(self, z_matrix, y):
        '''fit z-matrix in linear part to get LP'''
        reg = self.model.fit(z_matrix, y)
        
        LP = reg.predict(z_matrix)
        beta = reg.coef_       

        return LP, beta

    def compute(self, fdr=0.1):

        z_matrix = self.z_matrix
        plm_embedding = self.plm_embedding
        interaction_terms = self.interaction_terms
        y = self.y.copy()
        n, k, l = self.n, self.k, self.l

        # fit Z1_sig and Z2_plm interaction terms to y
        
        interaction_terms = interaction_terms.reshape(n,k*l)
        all_terms = np.concatenate([z_matrix, interaction_terms], axis=1)
        _, beta_all = self.fit_linear(all_terms, y)
        self.beta_all = beta_all.reshape(k, -1) # index_col 0 is Z1_sig standalone

        # Identify significant interaction terms

        rejections = self.filter_knockoffs(all_terms.reshape(n, -1), y, fdr=fdr)
        self.rejections = rejections

        beta_interaction = beta_all[k:].copy()
        sig_interaction = rejections[k:] * beta_interaction
        self.sig_interaction = sig_interaction.reshape(k,l)

        beta_interaction = beta_interaction.reshape(k,l)
        self.beta_interaction = beta_interaction

        return self.sig_interaction
    
    


            





