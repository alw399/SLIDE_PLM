import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os 

from statannotations.Annotator import Annotator
import itertools 

def show_interactions(machop, save_path=None):

    beta_interaction = machop.beta_interaction
    sig_interaction = machop.sig_mask * machop.sig_interaction

    index = machop.sig_LFs.copy()
    columns = list(range(machop.l))
    if not machop.interacts_only:
        index.append('null')
        columns[-1] = 'null'

    df = pd.DataFrame(beta_interaction, index=index, columns=columns)
    max_beta = np.max(np.abs(beta_interaction))

    df_sig = pd.DataFrame(sig_interaction, index=index, columns=columns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))

    # Plot beta_interaction
    sns.heatmap(data=df, square=True, ax=ax1, vmin=-max_beta, vmax=max_beta, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set(ylabel='LFs', xlabel='PLM embedding', title='Interaction Coefficients')

    # Plot sig_interaction
    sns.heatmap(data=df_sig, square=True, ax=ax2, vmin=0, vmax=1, 
                cmap='Blues', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
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


def show_performance(model, df, save_path=None):

    df = df.melt(id_vars='index', var_name='iter', value_name='auc')

    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    order = np.unique(df['index'])

    sns.boxplot(data=df, x='index', y='auc', hue='index', palette='hls', ax=ax, showfliers=False, order=order)
    sns.stripplot(data=df, x='index', y='auc', hue='index', ax=ax, palette='hls', legend=False, linewidth=1, edgecolor='black', jitter=True)

    pairs=list(itertools.combinations(order, 2))
    pairs = filter_pairs(pairs, df)

    annotator = Annotator(ax, pairs, data=df, x='index', y='auc', order=order)
    annotator.configure(test='Kruskal', text_format='star', loc='inside', verbose=2, hide_non_significant=True)
    annotator.apply_and_annotate()

    means = df.groupby('index')['auc'].mean()
    for i, mean in zip(means.index, means):
        plt.text(i, df['auc'].max() , f'Mean: {mean:.2f}', ha='center', va='bottom', fontsize=8, color='black')

    plt.title(f'{model} Performance')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


### Helper functions ###

def filter_pairs(pairs, df):
    filtered = []
    for i, j in pairs:
        if not np.all(df[df['index'] == i]['auc'].values == df[df['index'] == j]['auc'].values):
            filtered.append((i, j))
    return filtered


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

    # Plot beta_interaction
    sns.heatmap(data=df_sig, square=True, ax=ax1, vmin=-max_sig, vmax=max_sig, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set(ylabel='LFs', xlabel='PLM embedding', title=titles[0])

    # Plot sig_interaction
    sns.heatmap(data=df, square=True, ax=ax2, vmin=-max_beta, vmax=max_beta, 
                cmap='vlag', cbar_kws={'orientation': 'horizontal', 'shrink': 0.3})
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set(ylabel='LFs', xlabel='PLM embedding', title=titles[1])

    plt.tight_layout()


