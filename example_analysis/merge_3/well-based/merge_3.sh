#!/bin/bash
#SBATCH --job-name=merge_3
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=8:00:00
#SBATCH --output out/merge_3-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/aconcagua

# Create snakemake-output and hdf directories if they don't exist
mkdir -p merge_3/hdf

# Unlock
snakemake -s hash_3.smk --unlock

# Generate the rulegraph
snakemake -s hash_3.smk --rulegraph | dot -Tpdf > merge_3/hash_3.pdf

# Execute Snakemakes
snakemake -s hash_3.smk --use-conda --profile /lab/barcheese01/screens/aconcagua/merge_3/snakemake-configs --latency-wait 60 --rerun-incomplete

# Unlock
snakemake -s merge_3.smk --unlock

# Generate the rulegraph
snakemake -s merge_3.smk --rulegraph | dot -Tpdf > merge_3/merge_3.pdf

# Execute Snakemakes
snakemake -s merge_3.smk --use-conda --profile /lab/barcheese01/screens/aconcagua/merge_3/snakemake-configs --latency-wait 60 --rerun-incomplete