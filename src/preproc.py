import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

class DataProcessor:
    def __init__(self, rna_df, bcr_df, y_df):
        """
        Initialize the DataProcessor with DataFrames.
        :param rna_df: RNA DataFrame
        :param bcr_df: BCR DataFrame
        :param y_df: Target DataFrame
        """
        self.rna_df = rna_df
        self.bcr_df = bcr_df
        self.y_df = y_df

    def get_rna_metrics(self, df=None):
        """
        Compute and print basic metrics of the RNA DataFrame.
        :param df: DataFrame to compute metrics on
        :param filtered: Whether the DataFrame is filtered
        :return: None
        """
        if df is None:
            df = self.rna_df
            print('Computing for Raw RNA DataFrame')
            
        print('RNA DataFrame shape:', df.shape)
        print('Number of genes:', df.shape[1])
        print('Number of cells:', df.shape[0])
        print('Mean expression per gene:', df.mean(axis=0).mean())
        print('     Maximum expression per gene:', df.max(axis=0).max())
        print('     Minimum expression per gene:', df.min(axis=0).min())
        print('Mean expression per cell:', df.mean(axis=1).mean())
        print('     Maximum expression per cell:', df.max(axis=1).max())
        print('     Minimum expression per cell:', df.min(axis=1).min())

    def log_transform(self, df):
        """
        Apply log transformation to the given DataFrame.
        :param df: DataFrame to transform
        :return: Transformed DataFrame
        """
        return np.log1p(df)

    def var_filter(self, df, thresh=0.01):
        """
        Filter columns based on variance threshold.
        :param df: DataFrame to filter
        :param thresh: Variance threshold
        :return: Filtered DataFrame
        """
        variances = df.var(axis=0)
        filtered_df = df.loc[:, variances > thresh]
        print('Filtering genes by variance with thresh', thresh, ':', filtered_df.shape)
        return filtered_df

    def stdev_filter(self, df, thresh=0.01):
        """
        Filter columns based on standard deviation threshold.
        :param df: DataFrame to filter
        :param thresh: Standard deviation threshold
        :return: Filtered DataFrame
        """
        stdevs = df.std(axis=0)
        filtered_df = df.loc[:, stdevs > thresh]
        print('Filtering genes by standard deviation with thresh', thresh, ':', filtered_df.shape)
        return filtered_df

    def sparsity_filter_cells(self, df, thresh=0.1):
        """
        Filter rows based on sparsity threshold.
        :param df: DataFrame to filter
        :param thresh: Sparsity threshold (proportion of zeros in a row)
        :return: Filtered DataFrame
        """
        sparsity = df.apply(lambda x: np.count_nonzero(np.array(x) == 0) / len(x), axis=1)
        filtered_df = df[sparsity <= thresh]
        print('Filtering cells by sparsity with thresh', thresh, ':', filtered_df.shape)
        return filtered_df

    def filter_tcr_genes(self, df):
        """
        Remove T cell receptor genes (TRA, TRB, TRD, TRG) from the DataFrame.
        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        pattern = r'^TR[ABDG]'
        tcr_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(tcr_cols, axis=1)
        print('Filtering TCR genes:', filtered_df.shape)
        return filtered_df

    def filter_ig_genes(self, df):
        """
        Remove immunoglobulin genes (IGH, IGK, IGL) from the DataFrame.
        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        pattern = r'^IG[HKL]'
        ig_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(ig_cols, axis=1)
        print('Filtering IG genes:', filtered_df.shape)
        return filtered_df

    def filter_gm_genes(self, df):
        """
        Remove GM genes from the DataFrame.
        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        pattern = r'^GM'
        gm_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(gm_cols, axis=1)
        print('Filtering GM genes:', filtered_df.shape)
        return filtered_df

    def filter_mito_genes(self, df):
        """
        Remove mitochondrial genes from the DataFrame.
        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        pattern = r'^MT-'
        mito_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(mito_cols, axis=1)
        print('Filtering mitochondrial genes:', filtered_df.shape)
        return filtered_df

    def filter_ribo_genes(self, df):
        """
        Remove ribosomal protein genes from the DataFrame.
        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        pattern = r'^RPL|^RPS'
        ribo_cols = [s for s in df.columns if re.match(pattern, s, re.IGNORECASE)]
        filtered_df = df.drop(ribo_cols, axis=1)
        print('Filtering ribosomal genes:', filtered_df.shape)
        return filtered_df

    def filter_all_specific_genes(self, df):
        """
        Apply all specific gene filters (TCR, IG, GM, mitochondrial, ribosomal).
        :param df: DataFrame to filter
        :return: Filtered DataFrame
        """
        df = self.filter_tcr_genes(df)
        df = self.filter_ig_genes(df)
        df = self.filter_gm_genes(df)
        df = self.filter_mito_genes(df)
        df = self.filter_ribo_genes(df)
        print('Filtered all specific genes:', df.shape)
        return df
    
    def subset_bcr(self, df):
        """
        Subset BCR DataFrame to match the index of the given DataFrame.
        :param df: DataFrame to subset BCR DataFrame to
        :return: Subsetted BCR DataFrame
        """
        bcr_df = self.bcr_df.loc[df.index]
        print('Subset BCR DataFrame:', bcr_df.shape)
        return bcr_df
    
    def subset_y(self, df):
        """
        Subset target DataFrame to match the index of the given DataFrame.
        :param df: DataFrame to subset target DataFrame to
        :return: Subsetted target DataFrame
        """
        y_df = self.y_df.loc[df.index]
        print('Subset target DataFrame:', y_df.shape, Counter(y_df.iloc[:, 0]))
        return y_df
 