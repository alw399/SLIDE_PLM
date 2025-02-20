#############################################################################################################
# Running ESM-2 model to get sequence representations (650M model with 33 layers); code from www.github.com/facebookresearch/esm
# Input: dataframe with sequences, sequence column name, output folder name, output file name
# Output: numpy array with ESM2 sequence representations 
# example usage: python Run_ESM2.py -df data.csv -s sequence -o output -n ESM2_representations
# #############################################################################################################

import esm
import torch
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
parser.add_argument('-df','--df',help='dataframe with sequences',required=True)
parser.add_argument('-s','--s_col',help='sequence column name',required=True)
parser.add_argument('-n','--name',help='output file name',required=True)
args = parser.parse_args()

# load in data and sequences
df = pd.read_csv(args.df)
data = list(zip(df.index, df[args.s_col].values)) # (index, sequence)

# load ESM-2 model (650M model with 33 layers; https://github.com/facebookresearch/esm)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
token_representations = results["representations"][33] # 33 LAYERS

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# Convert each tensor to a numpy array and store in a list
array_list = [tensor.numpy() for tensor in sequence_representations]

# Stack the numpy arrays into a single numpy array
ESM2_array = np.stack(array_list)
print('ESM2 Embeddings shape:', ESM2_array.shape)

# Save sequence representations
np.save(args.name+'ESM2_embeddings', ESM2_array)

print('Done!')
