#!/bin/bash
#SBATCH --job-name=sbs_1
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --output out/sbs_1-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/denali

# Create snakemake-output and hdf directories if they don't exist
mkdir -p sbs_1/snakemake-output
mkdir -p sbs_1/hdf

# Unlock
snakemake -s sbs_1.smk --unlock

# Generate the rulegraph
snakemake -s sbs_1.smk --rulegraph | dot -Tpdf > sbs_1/sbs_1.pdf

# Execute Snakemake
snakemake -s sbs_1.smk --use-conda --profile /lab/barcheese01/screens/denali/sbs_1/snakemake-configs --latency-wait 60 --rerun-incomplete