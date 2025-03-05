import numpy as np 
import pandas as pd 
import scanpy as sc

import sys, os
sys.path.append('../src')

from interaction import Interaction
from util import compute_auc
from util import get_genes_from_slide_outs

from models import Estimator
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPClassifier

import seaborn as sns 
import matplotlib.pyplot as plt 
from statannotations.Annotator import Annotator
import itertools 


results_dir = '/ix/djishnu/alw399/SLIDE_PLM/results'

def filter_pairs(pairs, df):
    filtered = []
    for i, j in pairs:
        if not np.all(df[df['index'] == i]['auc'].values == df[df['index'] == j]['auc'].values):
            filtered.append((i, j))
    return filtered


runs_dict = {}

# JING CLONAL EXPANSION
name = 'jing_clonal_expansion'
x_path = '/ix/djishnu/Jane/SLIDESWING/jing_data/KIR+TEDDY/data/KIR+TEDDY_rna_filtered85.csv'
y_path = '/ix/djishnu/Jane/SLIDESWING/jing_data/KIR+TEDDY/data/KIR+TEDDY_Yexpanded_filtered85.csv'
slide_outs = '/ix/djishnu/Jane/SLIDESWING/jing_data/KIR+TEDDY/KIR+TEDDY_filtered85/KIR+TEDDY_filtered85_noint_output/0.01_0.5_out'
y = pd.read_csv(y_path)['Y'].values
species = 'human'
runs_dict[name] = {
    'x_path': x_path,
    'y_path': y_path,
    'slide_outs': slide_outs,
    'y': y,
    'species': species
}

# JING TUMOR TIL VS TEMRA
name='jing_tumor'
x_path = '/ix/djishnu/alw399/SLIDE_PLM/data/jing_tumor/tumor_x2.csv'
y_path = '/ix/djishnu/alw399/SLIDE_PLM/data/jing_tumor/tumor_y2.csv'
slide_outs = '/ix/djishnu/alw399/SLIDE_PLM/data/jing_tumor/0.05_0.5_out'
y = pd.read_csv(y_path)['y'].values
species = 'human'
runs_dict[name] = {
    'x_path': x_path,
    'y_path': y_path,
    'slide_outs': slide_outs,
    'y': y,
    'species': species
}

# ALOK ANTIGEN SPECIFICITY 
name = 'alok_antigen'
x_path = '/ix/djishnu/Jane/SLIDESWING/alok_data/data/Ins1_InsChg2_rna_MRfilt_forSLIDE.csv'
y_path = '/ix/djishnu/Jane/SLIDESWING/alok_data/data/Ins1_InsChg2_rna_MRfilt_antigens.csv'
slide_outs = '/ix/djishnu/Jane/SLIDESWING/alok_data/alok_data12_MRfilt_noint_out/0.01_2_out'
y = pd.read_csv(y_path)['Antigen'].values - 1
species = 'mouse'
runs_dict[name] = {
    'x_path': x_path,
    'y_path': y_path,
    'slide_outs': slide_outs,
    'y': y,
    'species': species
}
### START RUNS ###

for name, info in runs_dict.items():

    x_path = info['x_path']
    y_path = info['y_path']
    slide_outs = info['slide_outs']
    y = info['y']
    species = info['species']

    lf_dict = get_genes_from_slide_outs(slide_outs)
    all_genes = np.unique(np.concatenate([lf_dict[lf] for lf in lf_dict]))

    gene_embeddings = pd.read_csv(f'../data/ppi/{species}_embeddings.csv', index_col=0)
    null_genes = [g for g in all_genes if g not in gene_embeddings.index]
    print(f'Found {len(all_genes)-len(null_genes)}/{len(all_genes)} gene embeddings')

    gene_embeddings = pd.concat([
        gene_embeddings, 
        pd.DataFrame(
            index=null_genes, 
            columns=gene_embeddings.columns).fillna(1)
        ], axis=0)

    gene_embeddings = gene_embeddings.loc[all_genes]

    gex_df = pd.read_csv(x_path, usecols=list(all_genes))
    gex_threshes = gex_df.mean(axis=0)

    mask_df = pd.DataFrame(
        np.where(gex_df > gex_threshes, 1, 0), 
        index=gex_df.index, 
        columns=gex_df.columns
    )


    genept_df = np.einsum('ij,jk->ijk', mask_df.values, gene_embeddings)
    genept_df = genept_df.reshape(gex_df.shape[0], -1)
    wgenept_df = gex_df @ gene_embeddings


    z_matrix = pd.read_csv(os.path.join(slide_outs, 'z_matrix.csv'), index_col=0)
    z_matrix = z_matrix[list(lf_dict.keys())].values

    os.makedirs(f'{results_dir}/{name}', exist_ok=True)

    for model in [
        Lasso(alpha=0.05, max_iter=10000),
        LinearRegression(),
        MLPClassifier(max_iter=1000)
        ]:

        lasso0 = Estimator(model=model)
        auc0 = lasso0.evaluate(z_matrix, y)

        lasso1 = Estimator(model=model)
        auc1 = lasso0.evaluate(gex_df.values, y)

        lasso2 = Estimator(model=model)
        auc2 = lasso2.evaluate(mask_df.values, y)

        lasso3 = Estimator(model=model)
        auc3 = lasso3.evaluate(genept_df, y)

        lasso4 = Estimator(model=model)
        auc4 = lasso3.evaluate(wgenept_df.values, y)

        df = pd.DataFrame(
            np.vstack([auc0, auc1, auc2, auc3, auc4]),
            index=['z-matrix', 'gex', 'mask_gex', 'mask_ppi', 'w_ppi']
        )
        df.to_csv(f'{results_dir}/{name}/ppi_{model.__class__.__name__}.csv', index=False)
        df.reset_index(inplace=True)
        df = df.melt(id_vars='index', var_name='iter', value_name='auc')

        ### Start plotting ###
        fig, ax = plt.subplots(figsize=(10,10), dpi=150)

        sns.boxplot(data=df, x='index', y='auc', hue='index', palette='hls', ax=ax, showfliers=False, order=np.unique(df['index']))
        sns.stripplot(data=df, x='index', y='auc', hue='index', ax=ax, palette='hls', legend=False, linewidth=1, edgecolor='black', jitter=True)

        pairs=list(itertools.combinations(np.unique(df['index']), 2))
        pairs = filter_pairs(pairs, df)

        annotator = Annotator(ax, pairs, data=df, x='index', y='auc', order=np.unique(df['index']))
        annotator.configure(test='Kruskal', text_format='star', loc='inside', verbose=2, hide_non_significant=True)
        annotator.apply_and_annotate()

        means = df.groupby('index')['auc'].mean()
        for i, mean in zip(means.index, means):
            plt.text(i, df['auc'].max()+0.001 , f'Mean: {mean:.2f}', ha='center', va='bottom', fontsize=8, color='black')

        plt.title(f'{model.__class__.__name__} Performance')
        plt.savefig(f'{results_dir}/{name}/ppi_{model.__class__.__name__}.png')
        plt.tight_layout()
        plt.close()


