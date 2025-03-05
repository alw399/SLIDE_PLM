import numpy as np 
import pandas as pd 
import os, pickle
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, Lasso
from knockpy import KnockoffFilter

from util import get_sigLFs, compute_cohens_d, compute_auc

class Interaction():

    def __init__(
            self, slide_outs, 
            plm_embed, y, 
            z_matrix=None,
            interacts_only=False, 
            model='LR'
        ):

        self.slide_outs = slide_outs
        self.sig_LFs = get_sigLFs(slide_outs)
        self.interacts_only = interacts_only

        if interacts_only:
            self.plm_embedding = plm_embed
        else:
            self.plm_embedding = np.hstack([
                plm_embed, 
                np.ones((plm_embed.shape[0], 1))])

        self.y = y

        self.z_matrix = self.get_z_matrix(z_matrix, interacts_only=interacts_only)
        self.n, self.k = self.z_matrix.shape
        self.l = self.plm_embedding.shape[1] 

        self.interaction_terms = self.get_interaction_terms(self.z_matrix, self.plm_embedding)

        if model == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif model == 'LR':
            self.model = LinearRegression()
        else:
            raise ValueError('Model not supported')

    def get_z_matrix(self, z_matrix=None, interacts_only=True):
        if z_matrix is None:
            z_matrix = pd.read_csv(os.path.join(self.slide_outs, 'z_matrix.csv'), index_col=0)
            z_matrix = z_matrix[self.sig_LFs].values

        if not interacts_only:
            z_matrix = np.hstack([
                    z_matrix, 
                    np.ones((z_matrix.shape[0], 1))])

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
        interaction_terms = self.interaction_terms
        y = self.y.copy()
        n, k, l = self.n, self.k, self.l

        # fit Z1_sig and Z2_plm interaction terms to y
        
        interaction_terms = interaction_terms.reshape(n,k*l)
        
        if self.interacts_only:

            all_terms = np.concatenate([z_matrix, interaction_terms], axis=1)
            _, beta_all = self.fit_linear(all_terms, y)
            beta_all = beta_all.reshape(k, -1) # index_col 0 is Z1_sig standalone

            # Identify significant interaction terms

            rejections = self.filter_knockoffs(all_terms.reshape(n, -1), y, fdr=fdr)

            beta_interaction = beta_all[k:].copy()
            sig_interaction = rejections[k:] * beta_interaction
        
        else:
            
            _, beta_all = self.fit_linear(interaction_terms, y)
            beta_all = beta_all.reshape(k, -1)

            # Identify significant interaction terms

            rejections = self.filter_knockoffs(interaction_terms, y, fdr=fdr)

            beta_interaction = beta_all.reshape(-1)
            sig_interaction = rejections * beta_interaction


        sig_interaction = sig_interaction.reshape(k,l)
        sig_mask = np.where(sig_interaction != 0, 1, 0)       # save bc sig_interaction may be overwritten

        beta_interaction = beta_interaction.reshape(k,l)
        beta_interaction = beta_interaction

        return sig_mask, beta_interaction, sig_interaction

    def get_joint_embed(self):

        sig_mask = self.sig_mask.astype(bool)
        interaction_terms = self.interaction_terms[:, sig_mask].reshape(self.n, -1)
        coefs = self.beta_interaction[sig_mask]

        joint_embed = np.einsum('ij,j->ij', interaction_terms, coefs)
        self.joint_embed = joint_embed


    def get_sig_interactions(self, fdr=0.5, n_iters=10, thresh=0.4):
        '''
        Attributes:
        sig_interaction: Percentage of times an interaction term is significant across iterations.
        sig_mask: Binary mask indicating significant interactions based on the threshold.
        beta_interaction: Coefficients for the significant interaction terms.
        '''
        
        sig_interactions = []

        for i in tqdm(range(n_iters)):
            sig_mask, _, _ = self.compute(fdr=fdr)
            sig_interactions.append(sig_mask.copy())

        sig_interactions = np.stack(sig_interactions, axis=0)
        sig_interactions = np.mean(sig_interactions, axis=0)
        self.sig_interaction = sig_interactions
        self.sig_mask = np.where(sig_interactions > thresh, 1, 0)

        # Get the betas for the significant interactions
        interaction_terms = self.interaction_terms * self.sig_mask
        interaction_terms = interaction_terms.reshape(self.n, self.k*self.l)
        preds, beta_all = self.fit_linear(interaction_terms, self.y)

        self.beta_interaction = beta_all.reshape(self.k, self.l)
        self.beta_interaction = self.beta_interaction * self.sig_mask   # in case of underflow issues
        
        score = compute_auc(preds, self.y)
        print(f'Found {np.sum(self.sig_mask)} significant interactions with AUC={score}')


    def filter_effect_size(self):
        ys = np.unique(self.y)
        assert len(ys) == 2, 'y must be binary'

        group1z1 = self.z_matrix[self.y == ys[0]]
        group2z1= self.z_matrix[self.y == ys[1]]
        group1z2 = self.plm_embedding[self.y == ys[0]]
        group2z2 = self.plm_embedding[self.y == ys[1]]

        d_matrix = np.zeros((self.k, self.l))

        for z1_idx, z2_idx in zip(*np.where(self.beta_interaction != 0)):
            
            group1 = group1z1[:, z1_idx] * group1z2[:, z2_idx]
            group2 = group2z1[:, z1_idx] * group2z2[:, z2_idx]

            d = compute_cohens_d(group1, group2)
            d_matrix[z1_idx, z2_idx] = d
        
        self.d_matrix = d_matrix
        return d_matrix

    def filter_by_effect(self, threshold=0.1):
        self.sig_filtered = np.where(np.abs(self.d_matrix) > threshold, self.sig_interaction, 0)
    
    def force_linear_terms(self, z1=True, z2=True):
        assert self.interacts_only == False, 'Cannot force linear terms when they were not present'

        if z1: 
            # Force last column so that all Z1 terms come through
            self.sig_mask[:, -1] = 1

        if z2:
            # Prevent last row from being filtered out
            self.sig_mask[-1, :] = 1
        
        self.sig_mask[-1, -1] = 0 # This term is meaningless
        

        # Refit to get the new coefficients
        interaction_terms = self.interaction_terms * self.sig_mask
        interaction_terms = interaction_terms.reshape(self.n, self.k*self.l)
        preds, beta_all = self.fit_linear(interaction_terms, self.y)

        self.beta_interaction = beta_all.reshape(self.k, self.l)

        # in case of underflow issues
        self.beta_interaction = self.beta_interaction * self.sig_mask
        
        score = compute_auc(preds, self.y)
        print(f'Found {np.sum(self.sig_mask)} significant interactions with AUC={score}')




    




    
    


            





