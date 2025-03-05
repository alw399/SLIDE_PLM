#!/bin/bash
#SBATCH -t 3-00:00

#SBATCH --job-name=Run_ESM2_PCA
#SBATCH --mail-user=jcsiwe@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=150g
#SBATCH --cpus-per-task=16
#SBATCH --output=Run_ESM2_PCA.out

module load python/ondemand-jupyter-python3.9

python Run_ESM2_PCA.py -i ESM2_embeddings.npy -n TotalTeddy -d 16 32 64