#!/bin/bash
#SBATCH --job-name=preprocessing_0
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --output out/preprocessing_0-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/denali/

# Create snakemake-output and hdf directories if they don't exist
mkdir -p preprocessing_0/snakemake-output

# Generate the rulegraph
snakemake -s preprocessing_0.smk --rulegraph | dot -Tpdf > preprocessing_0/preprocessing_0.pdf

# Execute Snakemake
snakemake -s preprocessing_0.smk --use-conda --profile /lab/barcheese01/screens/denali/preprocessing_0/snakemake-configs --latency-wait 60