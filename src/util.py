from glob import glob
import pandas as pd
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
