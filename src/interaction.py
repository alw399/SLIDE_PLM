import numpy as np 
import pandas as pd 
import os

from sklearn.linear_model import LinearRegression
from knockpy import KnockoffFilter

from util import get_sigLFs

class Interaction():

    def __init__(self, slide_outs):
        self.slide_outs = slide_outs
        self.sig_LFs = get_sigLFs(slide_outs)

        self.z_matrix = self.get_z_matrix()
        self.plm_embedding = self.get_plm_embedding()
        self.y = self.get_y()

        self.n, self.k = self.z_matrix.shape
        self.l = self.plm_embedding.shape[1]

        self.interaction_terms = self.get_interaction_terms(self.z_matrix, self.plm_embedding)

        self.beta_interaction = None
        self.rejections = None

    def get_z_matrix(self):
        z_matrix = pd.read_csv(os.path.join(self.slide_outs, 'z_matrix.csv'), index_col=0)
        z_matrix = z_matrix[self.sig_LFs]
        return z_matrix.values

    def get_plm_embedding(self):
        print('Warning: hardcoded path to plm embedding')
        return np.load('/ix/djishnu/Jane/SLIDE_PLM/jing_expansion/KIR+CD8_testVAE_hidden_layer2.npy')
        
    def get_y(self):
        print('Warning: hardcoded path to y')
        return pd.read_csv('/ix/djishnu/Jane/SLIDESWING/jing_data/KIR+CD8/data/KIR+CD8_Yexpanded_filtered85.csv')['Y'].values

    @staticmethod
    def get_interaction_terms(z_matrix, plm_embedding):
        '''
        @return: interactions in shape of (n_samples, n_LFs, plm_embed_dim)
        '''
        return np.einsum('ij,ik->ijk', z_matrix, plm_embedding)

    @staticmethod
    def filter_knockoffs(interaction_terms, y):
        '''
        @return: mask of 0,1 significant interaction terms
        '''
        kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
        rejections = kfilter.forward(X=interaction_terms, y=y, fdr=0.1, shrinkage="ledoitwolf")
        return rejections
    
    @staticmethod
    def fit_linear(z_matrix, y):
        '''fit z-matrix in linear part to get LP'''
        reg = LinearRegression().fit(z_matrix, y)
        
        LP = reg.predict(z_matrix)
        beta = reg.coef_       

        return LP, beta

    def compute(self):

        z_matrix = self.z_matrix
        plm_embedding = self.plm_embedding
        interaction_terms = self.interaction_terms
        y = self.y.copy()
        n, k, l = self.n, self.k, self.l

        # fit Z1_sig and Z2_plm interaction terms to y
        
        interaction_terms = interaction_terms.reshape(n,k*l)
        all_terms = np.concatenate([z_matrix, interaction_terms], axis=1)
        _, beta_all = self.fit_linear(all_terms, y)

        # Identify significant interaction terms

        rejections = self.filter_knockoffs(all_terms.reshape(n, -1), y)
        self.rejections = rejections

        beta_interaction = beta_all[k:].copy()
        beta_interaction = beta_interaction.reshape(k,l)

        return beta_interaction




