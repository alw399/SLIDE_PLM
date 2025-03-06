from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import os


def get_sigLFs(slide_outs):
    '''
    Get the significant latent factors from the slide outputs
    @param slide_outs: path to the best performining slide outputs
    @return sigLFs: list of significant latent factors
    '''
    sig_LFs = glob(str(Path(slide_outs) / '*_list*'))
    sig_LFs = [f"Z{path.replace('.txt','').rsplit('Z', 1)[1]}" for path in sig_LFs]

    return sig_LFs


def get_genes_from_slide_outs(slide_outs):
    gene_lists = glob(f'{slide_outs}/*_list*')
    genes = {}

    for path in gene_lists:
        lf = os.path.basename(path).split('_')[-1].split('.')[0]
        genes[lf] = list(pd.read_csv(path, sep='\t')['names'])
    
    return genes

def compute_auc(yhat, y):
    '''
    Compute the AUC score
    @param yhat: predicted values
    @param y: true values
    @return auc: AUC score
    '''
    from sklearn.metrics import roc_auc_score
    yhat = [1 if i >= 0.5 else 0 for i in yhat]
    auc = roc_auc_score(y, yhat)
    return auc

def remove_empty_tcrs(sequences, y=None, z_matrix=None, null_value='-', remove_nans=True):

    if not isinstance(sequences, np.ndarray):
        sequences = sequences.values

    if remove_nans:
        sequences = np.where(pd.isna(sequences), null_value, sequences)

    nulls = np.where(sequences == null_value)[0]
    sequences = np.delete(sequences, nulls)

    if y is not None:
        y = np.delete(y, nulls)
    
    if z_matrix is not None:
        # z_matrix = np.delete(z_matrix, nulls, axis=0)
        z_matrix = z_matrix.drop(z_matrix.index[nulls], axis=0)

    return sequences, y, z_matrix


def compute_cohens_d(group1, group2):
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    # Calculate standard deviations
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    
    # Calculate sample sizes
    n1 = len(group1)
    n2 = len(group2)
    
    # Calculate pooled standard deviation
    sp = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / sp
    return d
