#!/bin/bash
#SBATCH --job-name=aggregate_4_eval
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output out/aggregate_4_eval-%j.out

# Load any necessary modules or activate virtual environments
source /lab/barcheese01/mdiberna/OpticalPooledScreens/venv_ops_new/bin/activate

# Change to the directory where your Snakefiles are located
cd /lab/barcheese01/screens/aconcagua

# Execute Snakemake
python3 aggregate_4/aggregate_4_eval.py