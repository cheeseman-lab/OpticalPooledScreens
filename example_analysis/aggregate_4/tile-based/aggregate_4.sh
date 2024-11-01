#!/bin/bash
#SBATCH --job-name=aggregate_4
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output out/aggregate_4-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/denali

# Create csv and hdf directories if they don't exist
mkdir -p aggregate_4/hdf
mkdir -p aggregate_4/csv
mkdir -p montage

# Unlock
snakemake -s aggregate_4.smk --unlock

# Generate the rulegraph
snakemake -s aggregate_4.smk --rulegraph | dot -Tpdf > aggregate_4/aggregate_4.pdf

# Execute Snakemakes
snakemake -s aggregate_4.smk --use-conda --profile /lab/barcheese01/screens/denali/aggregate_4/snakemake-configs --latency-wait 60 --rerun-incomplete