#############################################################################################################
# Running PCA on ESM2 sequence representations
# Input: numpy array with ESM2 sequence representations
# Output: PCA embeddings
# example usage: python Run_ESM2_PCA.py -i ESM2_representations.npy -n TotalTeddy -d 16 32 64
# #############################################################################################################

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
parser.add_argument('-i','--input',help='numpy array of esm seqeunces',required=True)
parser.add_argument('-n','--name',help='output file name',required=True)
parser.add_argument('-d','--dims',help='list of dimensions', nargs='+', type=int, required=False)
args = parser.parse_args()

# load in data
esm2_vecs = np.load(args.input, allow_pickle=True)
print('ESM2 Embeddings shape:', esm2_vecs.shape)

# Run PCA
for dim in args.dims:
    pca = PCA(n_components=dim, random_state=42)
    ESM2_PCA = pca.fit_transform(esm2_vecs)
    print('PCA shape:', ESM2_PCA.shape)
    np.save(args.name+'ESM2_PCA_'+str(dim), ESM2_PCA)

print('Done!')
