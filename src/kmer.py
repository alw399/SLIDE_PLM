import numpy as np 
import pandas as pd
import itertools

class Kmerizer:
    def __init__(self, k=2, token_size=8):
        self.k = k
        self.token_size = token_size

        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
                    'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
        self.lookup_table = self.create_lookup_table(self.amino_acids, self.k, self.token_size)

    @staticmethod
    def create_lookup_table(amino_acids, k=2, token_size=8):
        pairs = [''.join(p) for p in itertools.product(amino_acids, repeat=k)]
        pairs = np.array(pairs + ['X'])         # add padding token
        lookup_table = np.random.rand(pairs.shape[0], token_size)
        lookup_df = pd.DataFrame(lookup_table, index=pairs)
        return lookup_df
    
    def tokenize_kmers(self, tcr, max_len=None):
        k = self.k
        tokens = []
        for i in range(0, len(tcr) - k + 1):
            tokens.append(tcr[i:i+k])
        if max_len: # zero pad
            tokens += ['X'] * (max_len - len(tokens))
        return tokens

    def encode(self, tcr, max_len=None):
        tokens = self.tokenize_kmers(tcr, max_len)
        token_vectors = []
        for token in tokens:
            token_vectors.append(self.lookup_table.loc[token].values)
        return np.array(token_vectors)

    def encode_batch(self, tcrs):
        max_len = max([len(tcr) for tcr in tcrs])
        return np.array([self.encode(tcr, max_len) for tcr in tcrs])


