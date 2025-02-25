import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

def show_interactions(machop):

    sig_interaction = machop.sig_interaction
    beta_interaction = machop.beta_interaction

    index = machop.sig_LFs.copy()
    columns = list(range(machop.l))
    if not machop.interacts_only:
        index.append('null')
        columns[-1] = 'null'

    df = pd.DataFrame(beta_interaction, index=index, columns=columns)
    max_beta = np.max(np.abs(beta_interaction))

    df_sig = pd.DataFrame(sig_interaction, index=index, columns=columns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))

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

    plt.tight_layout()



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


