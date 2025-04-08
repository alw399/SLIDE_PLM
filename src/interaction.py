import numpy as np 
import pandas as pd 
import os, pickle
# from tqdm import tqdm
import enlighten

from sklearn.linear_model import LinearRegression, Lasso
from knockpy import KnockoffFilter

from util import get_sigLFs, compute_cohens_d, compute_auc

class Interaction():

    def __init__(
            self, slide_outs, 
            plm_embed, y, 
            z_matrix=None,
            interacts_only=False, 
            model='LR',
            name='Model'
        ):
        
        self.name = name
        self.slide_outs = slide_outs
        self.interacts_only = interacts_only

        if interacts_only:
            self.plm_embedding = plm_embed
        else:
            self.plm_embedding = np.hstack([
                plm_embed, 
                np.ones((plm_embed.shape[0], 1))])

        self.y = y

           
        if z_matrix is None:
            self.sig_LFs = get_sigLFs(slide_outs)
        else:
            self.sig_LFs = list(z_matrix.columns)

        self.z_matrix = self.get_z_matrix(
            z_matrix, interacts_only=interacts_only)
    
        
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
            z_matrix = z_matrix[self.sig_LFs]

        if not interacts_only:
            n_obs = z_matrix.shape[0]
            z_matrix = np.hstack([
                    z_matrix.values.reshape(n_obs, -1), 
                    np.ones((n_obs, 1))])

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

    def compute(self, fdr=0.1, permuted=False):
        if permuted:
            z_matrix = self.z_matrix_perm
            interaction_terms = self.interaction_terms_perm
        else:
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

    def get_joint_embed_perm(self):

        sig_mask_perm = self.sig_mask_perm.astype(bool)
        interaction_terms_perm = self.interaction_terms_perm[:, sig_mask_perm].reshape(self.n, -1)
        coefs_perm = self.beta_interaction_perm[sig_mask_perm]

        joint_embed_perm = np.einsum('ij,j->ij', interaction_terms_perm, coefs_perm)
        self.joint_embed_perm = joint_embed_perm
    
    def get_sig_interactions(self, fdr=0.5, n_iters=10, thresh=0.4):
        '''
        Attributes:
        sig_interaction: Percentage of times an interaction term is significant across iterations.
        sig_mask: Binary mask indicating significant interactions based on the threshold.
        beta_interaction: Coefficients for the significant interaction terms.
        '''
        print('Computing sig interactions using ORIGINAL interactions')
        self.params = {
            'fdr': fdr,
            'n_iters': n_iters,
            'thresh': thresh
        }
        
        sig_interactions = []

        manager = enlighten.get_manager()
        pbar = manager.counter(
            total=n_iters, 
            desc='Computing interactions', 
            unit='iter', 
            color='pink', 
            autorefresh=True
        )

        for i in range(n_iters):
            sig_mask, _, _ = self.compute(fdr=fdr, permuted=False)
            sig_interactions.append(sig_mask.copy())

            pbar.desc = f'Iteration {i}: Found {sig_mask.sum()} significant interactions'
            pbar.refresh()
            pbar.count = i+1

        pbar.close()

        # for i in tqdm(range(n_iters)):
        #     sig_mask, _, _ = self.compute(fdr=fdr)
        #     sig_interactions.append(sig_mask.copy())
        

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
        self.score = score
        self.n_sig_interactions = np.sum(self.sig_mask)
        print(f'Found {np.sum(self.sig_mask)} significant interactions with AUC={score} (ORIGINAL)')

    def get_sig_interactions_permuted(self, fdr=0.5, n_iters=10, thresh=0.4):
        '''
        Attributes:
        sig_interaction: Percentage of times an interaction term is significant across iterations.
        sig_mask: Binary mask indicating significant interactions based on the threshold.
        beta_interaction: Coefficients for the significant interaction terms.
        '''
        print('Computing sig interactions using PERMUTED interactions')
        self.get_permuted_interactions(fdr=fdr, n_iters=n_iters, thresh=thresh)
        
        self.params = {
            'fdr': fdr,
            'n_iters': n_iters,
            'thresh': thresh
        }
        
        sig_interactions_perm = []

        manager = enlighten.get_manager()
        pbar = manager.counter(
            total=n_iters, 
            desc='Computing interactions', 
            unit='iter', 
            color='pink', 
            autorefresh=True
        )
            
        for i in range(n_iters):
            sig_mask_perm, _, _ = self.compute(fdr=fdr, permuted=True)
            sig_interactions_perm.append(sig_mask_perm.copy())

            pbar.desc = f'Iteration {i}: Found {sig_mask_perm.sum()} significant interactions'
            pbar.refresh()
            pbar.count = i+1

        pbar.close()

        sig_interactions_perm = np.stack(sig_interactions_perm, axis=0)
        sig_interactions_perm = np.mean(sig_interactions_perm, axis=0)
        self.sig_interaction_perm = sig_interactions_perm
        self.sig_mask_perm = np.where(sig_interactions_perm > thresh, 1, 0)

        # Get the betas for the significant interactions
        interaction_terms_perm  = self.interaction_terms_perm  * self.sig_mask_perm 
        interaction_terms_perm  = interaction_terms_perm .reshape(self.n, self.k*self.l)
        preds_perm , beta_all_perm  = self.fit_linear(interaction_terms_perm, self.y)

        self.beta_interaction_perm  = beta_all_perm .reshape(self.k, self.l)
        self.beta_interaction_perm  = self.beta_interaction_perm  * self.sig_mask_perm  # in case of underflow issues
        
        score_perm  = compute_auc(preds_perm, self.y)
        self.score_perm  = score_perm 
        self.n_sig_interactions_perm  = np.sum(self.sig_mask_perm)
        print(f'Found {np.sum(self.sig_mask_perm )} significant interactions with AUC={score_perm} (PERMUTED)')
        
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
        
    def get_permuted_interactions(self, fdr=0.5, n_iters=10, thresh=0.4):
        '''
        Mismatching the TCR and PLM embeddings to get a null distribution of interactionss
        '''
        print('Computing permuted interactions')
        zs = self.z_matrix.copy() # n x k
        plms = self.plm_embedding.copy() # n x l
        
        # shuffle samples
        zs_permuted = zs[np.random.permutation(zs.shape[0])]
        plms_permuted = plms[np.random.permutation(plms.shape[0])]
        self.z_matrix_perm = zs_permuted
        self.plm_embedding_perm = plms_permuted
        
        # fit Z1_sig and Z2_plm interaction terms to y
        self.interaction_terms_perm = self.get_interaction_terms(zs_permuted, plms_permuted)
        
                
        

        
        
       
