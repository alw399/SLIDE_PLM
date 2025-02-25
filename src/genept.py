import numpy as np 
import pandas as pd 
import os 
from glob import glob
import pickle, json

class GenePTEmbedder():
    def __init__(self, species='mouse'):
        self.path = '/ix/djishnu/alw399/SLIDE_PLM/data/GenePT_emebdding_v2'
        
        gene2embed_path = os.path.join(self.path, 'GenePT_gene_embedding_ada_text.pickle')
        with open(gene2embed_path, 'rb') as f:
            self.gene2embed = pickle.load(f)
        
        self.blank = np.ones(1536)

        gene2summary_path = os.path.join(self.path, 'NCBI_summary_of_genes.json')
        with open(gene2summary_path, 'r') as f:
            self.gene2summary = json.load(f)

        self.fix_names(species)
    
    def fix_names(self, species):
        if species == 'mouse':
            gene2embed = {k.capitalize(): v for k, v in self.gene2embed.items()}
            gene2summary = {k.capitalize(): v for k, v in self.gene2summary.items()}
        
            self.gene2embed = gene2embed
            self.gene2summary = gene2summary

    def get_gene_embedding(self, gene):
        return np.array(self.gene2embed[gene])
    
    def get_gene_summary(self, gene):
        return self.gene2summary[gene]

    def get_gene_info(self, genes, embedding=True):
        overlap = len(set(genes) & set(self.gene2embed.keys()))
        if overlap - len(set(genes)) < 0:
            print(f'Found {overlap}/{len(genes)} embeddings')

        if embedding:
            return np.array([self.gene2embed.get(gene, self.blank) for gene in genes])
        else:
            return [self.gene2summary.get(gene, self.blank) for gene in genes]