import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os 

from statannotations.Annotator import Annotator
import itertools 

def show_interactions(machop, save_path=None, z1=None, z2=None):

    beta_interaction = machop.beta_interaction
    sig_interaction = machop.sig_interaction

    if z2 is None: z2 = 'PLM embedding'
    if z1 is None: z1 = 'LFs'

    index = machop.sig_LFs.copy()
    columns = list(range(machop.l))
    if not machop.interacts_only:
        index.append('null')
        columns[-1] = 'null'

    df = pd.DataFrame(beta_interaction, index=index, columns=columns)
    max_beta = np.max(np.abs(beta_interaction))

    df_sig = pd.DataFrame(sig_interaction, index=index, columns=columns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))

    if machop.interacts_only:
        add = 0
    else:
        add = 1
    
    if machop.version == 'interaction':
        x_labels = list(range(machop.l)) + ['null']*add
    elif machop.version == 'standard_ridge_interaction' or machop.version == 'bayesian_ridge_interaction':
        x_labels = list(machop.plm_subset_idx) + ['null']*add
    else:
        raise ValueError(f'Unknown version: {machop.version}')
    
    # Plot beta_interaction
    sns.heatmap(data=df, square=True, ax=ax1, vmin=-max_beta, vmax=max_beta, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set(
        ylabel=z1, xlabel=z2, 
        title='Interaction Coefficients: '+str(machop.n_sig_interactions)+' significant interactions, AUC='+str(round(machop.score,5))
    )
    # print(f'Found {np.sum(self.sig_mask)} significant interactions with AUC={score}')

    # Plot sig_interaction
    sns.heatmap(data=df_sig, square=True, ax=ax2, vmin=0, vmax=1, 
                cmap='Blues', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set(ylabel=z1, xlabel=z2, title='Percentage of times significant')

    # Indicate nonzero values
    for i in range(df.shape[0]):
        for j in range(df_sig.shape[1]):
            if df.iloc[i, j] != 0:
                ax1.text(j + 0.5, i + 0.5, f'{df.iloc[i, j]:.2f}', 
                         ha='center', va='center', color='black')
            
            if df_sig.iloc[i, j] != 0:
                ax2.text(j + 0.5, i + 0.5, f'{df_sig.iloc[i, j]:.2f}', 
                         ha='center', va='center', color='black')
            

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

def show_interactions_perm(machop, save_path=None):

    beta_interaction_perm = machop.beta_interaction_perm
    sig_interaction_perm = machop.sig_interaction_perm

    index = machop.sig_LFs.copy()
    columns = list(range(machop.l))
    if not machop.interacts_only:
        index.append('null')
        columns[-1] = 'null'

    df = pd.DataFrame(beta_interaction_perm, index=index, columns=columns)
    max_beta = np.max(np.abs(beta_interaction_perm))

    df_sig = pd.DataFrame(sig_interaction_perm, index=index, columns=columns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))

    if machop.interacts_only:
        add = 0
    else:
        add = 1
    
    if machop.version == 'interaction':
        x_labels = list(range(machop.l)) + ['null']*add
    elif machop.version == 'standard_ridge_interaction' or machop.version == 'bayesian_ridge_interaction':
        x_labels = list(machop.plm_subset_idx) + ['null']*add
    else:
        raise ValueError(f'Unknown version: {machop.version}')
    
    # Plot beta_interaction
    sns.heatmap(data=df, square=True, ax=ax1, vmin=-max_beta, vmax=max_beta, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set(ylabel='LFs', xlabel='PLM embedding', title='Interaction Coefficients: '+str(machop.n_sig_interactions_perm)+' significant interactions, Permuted AUC='+str(round(machop.score_perm,5)))
    # print(f'Found {np.sum(self.sig_mask)} significant interactions with AUC={score}')

    # Plot sig_interaction
    sns.heatmap(data=df_sig, square=True, ax=ax2, vmin=0, vmax=1, 
                cmap='Blues', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set(ylabel='LFs', xlabel='PLM embedding', title='Percentage of times significant')

    # Indicate nonzero values
    for i in range(df.shape[0]):
        for j in range(df_sig.shape[1]):
            if df.iloc[i, j] != 0:
                ax1.text(j + 0.5, i + 0.5, f'{df.iloc[i, j]:.2f}', 
                         ha='center', va='center', color='black')
            
            if df_sig.iloc[i, j] != 0:
                ax2.text(j + 0.5, i + 0.5, f'{df_sig.iloc[i, j]:.2f}', 
                         ha='center', va='center', color='black')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

def show_performance(model, df, save_path=None, figsize=(6,6), order=None):

    df = df.melt(id_vars='index', var_name='iter', value_name='auc')

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    if order is None:
        order = np.unique(df['index'])

    sns.boxplot(data=df, x='index', y='auc', hue='index', palette='hls', ax=ax, showfliers=False, order=order)
    sns.stripplot(data=df, x='index', y='auc', hue='index', ax=ax, palette='hls', legend=False, linewidth=1, edgecolor='black', jitter=True)

    pairs=list(itertools.combinations(order, 2))
    pairs = filter_pairs(pairs, df)

    if pairs is not None:
        annotator = Annotator(ax, pairs, data=df, x='index', y='auc', order=order)
        annotator.configure(test='Kruskal', text_format='star', loc='inside', verbose=2, hide_non_significant=True)
        annotator.apply_and_annotate()

    means = df.groupby('index')['auc'].mean()
    for i, mean in zip(means.index, means):
        plt.text(i, df['auc'].max()*1.05 , f'Mean: {mean:.2f}', ha='center', va='bottom', fontsize=6, color='black')

    plt.title(f'{model} Performance')
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


### Helper functions ###

def filter_pairs(pairs, df):
    try:
        filtered = []
        for i, j in pairs:
            if not np.all(df[df['index'] == i]['auc'].values == df[df['index'] == j]['auc'].values):
                filtered.append((i, j))
        return filtered
    except:
        return None


### Outdated functions ###

def show_effect_sizes(machop):

    plot_double_heatmap(
        machop,
        machop.d_matrix, 
        machop.sig_interaction, 
        titles = ['Significant Interactions', 'Cohen\'s d Values']
    )

def show_filtered(machop):

    plot_double_heatmap(
        machop,
        machop.sig_filtered,
        machop.sig_interaction,
        titles = ['Significant Interactions', 'Filtered Interactions']
    )


def plot_double_heatmap(machop, beta_interaction, sig_interaction,
        titles=['Significant Interaction Coefficients', 'Interaction Coefficients']):

    index = machop.sig_LFs.copy()
    columns = list(range(machop.l))
    if not machop.interacts_only:
        index.append('null')
        columns[-1] = 'null'

    df = pd.DataFrame(beta_interaction, index=index, columns=columns)
    max_beta = min(np.max(np.abs(beta_interaction)), 1)

    df_sig = pd.DataFrame(sig_interaction, index=index, columns=columns)
    max_sig = np.max(np.abs(sig_interaction))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))

    if machop.interacts_only:
        add = 0
    else:
        add = 1
    
    if machop.version == 'interaction':
        x_labels = list(range(machop.l)) + ['null']*add
    elif machop.version == 'standard_ridge_interaction' or machop.version == 'bayesian_ridge_interaction':
        x_labels = list(machop.plm_subset_idx) + ['null']*add
    else:
        raise ValueError(f'Unknown version: {machop.version}')
    
    # Plot beta_interaction
    sns.heatmap(data=df_sig, square=True, ax=ax1, vmin=-max_sig, vmax=max_sig, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set(ylabel='LFs', xlabel='PLM embedding', title=titles[0])

    # Plot sig_interaction
    sns.heatmap(data=df, square=True, ax=ax2, vmin=-max_beta, vmax=max_beta, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set(ylabel='LFs', xlabel='PLM embedding', title=titles[1])

    plt.tight_layout()


