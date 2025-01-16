#!/bin/bash
#SBATCH --job-name=ph_2
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --output out/ph_2-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/denali

# Create snakemake-output and hdf directories if they don't exist
mkdir -p ph_2/snakemake-output
mkdir -p ph_2/hdf

# Unlock
snakemake -s ph_2.smk --unlock

# Generate the rulegraph
snakemake -s ph_2.smk --rulegraph | dot -Tpdf > ph_2/ph_2.pdf

# Execute Snakemake
snakemake -s ph_2.smk --use-conda --profile /lab/barcheese01/screens/denali/ph_2/snakemake-configs --latency-wait 60 --rerun-incomplete