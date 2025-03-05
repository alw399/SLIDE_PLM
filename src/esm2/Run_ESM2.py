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
import argparse

def load_data(file_path, sequence_column):
    df = pd.read_csv(file_path)
    data = list(zip(df.index, df[sequence_column].values))  # (index, sequence)
    return data

def load_model(use_torch=False):

    if use_torch:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    return model, alphabet, batch_converter

def get_representations(data, model, alphabet, batch_converter):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]  # 33 LAYERS

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    array_list = [tensor.numpy() for tensor in sequence_representations]
    ESM2_array = np.stack(array_list)
    return ESM2_array

def save_representations(array, output_name):
    np.save(output_name + 'ESM2_embeddings', array)

def main():
    parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
    parser.add_argument('-df', '--df', help='dataframe with sequences', required=True)
    parser.add_argument('-s', '--s_col', help='sequence column name', required=True)
    parser.add_argument('-n', '--name', help='output file name', required=True)
    args = parser.parse_args()

    data = load_data(args.df, args.s_col)
    model, alphabet, batch_converter = load_model()
    ESM2_array = get_representations(data, model, alphabet, batch_converter)
    print('ESM2 Embeddings shape:', ESM2_array.shape)
    save_representations(ESM2_array, args.name)
    print('Done!')

if __name__ == "__main__":
    main()
