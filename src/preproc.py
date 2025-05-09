import numpy as np
import re
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, rna_df, bcr_df, y_df):
        """
        Initialize the DataProcessor with a DataFrame.
        :param df: pandas DataFrame
        :param bcr_df: BCR DataFrame
        :param y_df: Target DataFrame
        """
        self.rna_df = rna_df
        self.bcr_df = bcr_df
        self.y_df = y_df

    def get_rna_metrics(self):
        """
        Compute and print basic metrics of the RNA DataFrame.
        :return: None
        """
        print('RNA DataFrame shape:', self.rna_df.shape)
        print('Number of genes:', self.rna_df.shape[1])
        print('Number of cells:', self.rna_df.shape[0])
        print('Number of non-zero entries:', np.count_nonzero(self.rna_df))
        print('Mean expression per gene:', self.rna_df.mean(axis=0).mean())
        print('Maximum expression per gene:', self.rna_df.max(axis=0).max())
        print('Minimum expression per gene:', self.rna_df.min(axis=0).min())
        print('Mean expression per cell:', self.rna_df.mean(axis=1).mean())
        print('Maximum expression per cell:', self.rna_df.max(axis=1).max())
        print('Minimum expression per cell:', self.rna_df.min(axis=1).min())
        
    def log_transform(self):
        """
        Apply log transformation to the RNA DataFrame.
        :return: Log-transformed DataFrame
        """
        df_log = np.log1p(self.rna_df)
        return df_log

    def var_filter(self, thresh=0.01):
        """
        Filter columns based on variance threshold.
        :param thresh: Variance threshold
        :return: Filtered DataFrame
        """
        variances = self.rna_df.var(axis=0)  # Compute variance per column
        df_filtered = self.rna_df.loc[:, variances > thresh]  # Keep only columns above threshold
        print('Filtering genes by variance with thresh '+str(thresh) +':', df_filtered.shape)
        return df_filtered

    def stdev_filter(self, thresh=0.01):
        """
        Filter columns based on standard deviation threshold.
        :param thresh: Standard deviation threshold
        :return: Filtered DataFrame
        """
        stdevs = self.rna_df.std(axis=0)  # Compute standard deviation per column
        df_filtered = self.rna_df.loc[:, stdevs > thresh]  # Keep only columns above threshold
        print('Filtering genes by standard deviation with thresh '+str(thresh) +':', df_filtered.shape)
        return df_filtered

    def sparsity_filter_cells(self, thresh=0.1):
        """
        Filter rows based on sparsity threshold.
        :param thresh: Sparsity threshold (proportion of zeros in a row)
        :return: Filtered DataFrame
        """
        sparsity = self.rna_df.apply(lambda x: np.count_nonzero(np.array(x) == 0) / len(x), axis=1)  # Compute sparsity per row
        df_filtered = self.rna_df[sparsity <= thresh]  # Keep only rows with sparsity <= threshold
        print('Filtering cells by sparsity with thresh '+str(thresh) +':', df_filtered.shape)
        return df_filtered

    def filter_tcr_genes(self):
        """
        Remove T cell receptor genes (TRA, TRB, TRD, TRG) from the DataFrame.
        :return: Filtered DataFrame
        """
        pattern = r'^TR[ABDG]'  # TRA, TRB, TRD, TRG (T cell receptor genes)
        tcr_cols = [s for s in self.rna_df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = self.rna_df.drop(tcr_cols, axis=1)
        print('Filtering TCR genes:', filtered_df.shape)
        return filtered_df

    def filter_ig_genes(self):
        """
        Remove immunoglobulin genes (IGH, IGK, IGL) from the DataFrame.
        :return: Filtered DataFrame
        """
        pattern = r'^IG[HKL]'  # IGH, IGK, IGL (Immunoglobulin genes)
        ig_cols = [s for s in self.rna_df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = self.rna_df.drop(ig_cols, axis=1)
        print('Filtering IG genes:', filtered_df.shape)
        return filtered_df

    def filter_gm_genes(self):
        """
        Remove GM genes from the DataFrame.
        :return: Filtered DataFrame
        """
        pattern = r'^GM'  # GM genes
        gm_cols = [s for s in self.rna_df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = self.rna_df.drop(gm_cols, axis=1)
        print('Filtering GM genes:', filtered_df.shape)
        return filtered_df

    def filter_mito_genes(df):
        pattern = r'^MT-' # Mitochondrial genes
        mito_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(mito_cols,axis=1)
        print('Filtering mitochondrial genes:', filtered_df.shape)
        return filtered_df
    
    def filter_ribo_genes(df):
        pattern = r'^RPL|^RPS' # Ribosomal protein genes
        ribo_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(ribo_cols,axis=1)
        print('Filtering ribosomal genes:', filtered_df.shape)
        return filtered_df
    
    def plot_row_sparsity_histogram(self, bins=20, title="Row Sparsity Histogram"):
        """
        Plot a histogram of row sparsity (proportion of zeros in each row).
        :param bins: Number of bins for the histogram
        :param title: Title of the histogram
        """
        row_sparsity = self.rna_df.apply(lambda x: np.count_nonzero(np.array(x) == 0) / len(x), axis=1)
        plt.figure(figsize=(8, 6))
        plt.hist(row_sparsity, bins=bins, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("Row Sparsity (Proportion of Zeros)")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    def subset_bcr_df(self):
        """
        Subset the BCR DataFrame to match the indices of the RNA DataFrame.
        :return: Subsetted BCR DataFrame
        """
        bcr_subset = self.bcr_df[self.bcr_df.index.isin(self.rna_df.index)]
        print('Subsetting BCR DataFrame:', bcr_subset.shape)
        return bcr_subset
    
    def subset_y_df(self):
        """
        Subset the target DataFrame to match the indices of the RNA DataFrame.
        :return: Subsetted target DataFrame
        """
        y_subset = self.y_df[self.y_df.index.isin(self.rna_df.index)]
        print('Subsetting target DataFrame:', y_subset.shape)
        return y_subset