#!/bin/bash
#SBATCH -t 3-00:00

#SBATCH --job-name=Run_ESM2
#SBATCH --mail-user=jcsiwe@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=150g
#SBATCH --cpus-per-task=16
#SBATCH --output=/ix/djishnu/Jane/SLIDE_PLM/ESM2/Run_ESM2.out

cd /ix/djishnu/Jane/SLIDE_PLM/

module load python/ondemand-jupyter-python3.9
source activate esm2

python Run_ESM2.py -df data.csv -s sequence -n TotalTeddy