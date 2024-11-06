#!/bin/bash
#SBATCH --job-name=cluster_5
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output out/cluster_5-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/denali

# Execute Snakemake
python3 cluster_5/cluster_5.py