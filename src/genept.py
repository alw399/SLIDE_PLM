import numpy as np 
import pandas as pd 
import os 
from glob import glob
import pickle, json

class GenePTEmbedder():
    def __init__(self):
        self.path = '/ix/djishnu/alw399/SLIDE_PLM/data/GenePT_emebdding_v2'
        
        gene2embed_path = os.path.join(self.path, 'GenePT_gene_embedding_ada_text.pickle')
        with open(gene2embed_path, 'rb') as f:
            self.gene2embed = pickle.load(f)
        

        gene2summary_path = os.path.join(self.path, 'NCBI_summary_of_genes.json')
        with open(gene2summary_path, 'r') as f:
            self.gene2summary = json.load(f)
        
    def get_gene_embedding(self, gene):
        return np.array(self.gene2embed[gene])
    
    def get_gene_summary(self, gene):
        return self.gene2summary[gene]

    def get_gene_info(self, genes, embedding=True):
        if embedding:
            return np.array([self.gene2embed[gene] for gene in genes if gene in self.gene2embed])
        else:
            return [self.gene2summary[gene] for gene in genes if gene in self.gene2summary]