#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ops.qc import *

# Set screen directories
ph_function_home = "/lab/barcheese01/screens"
ph_function_dataset = "denali"
home = os.path.join(ph_function_home, ph_function_dataset)

# Create plots directory if it doesn't exist
qc_dir = os.path.join(home, 'ph_2', 'qc')
os.makedirs(qc_dir, exist_ok=True)

def read_hdf_generator(file_pattern):
    for file in glob(file_pattern):
        yield pd.read_hdf(file)

def save_plot(fig, filename):
    fig.savefig(os.path.join(qc_dir, filename))
    plt.close(fig)

def concatenate_and_save(file_pattern, output_filename, key):
    data = pd.concat(read_hdf_generator(file_pattern), ignore_index=True)
    data.to_hdf(os.path.join(home, 'ph_2', 'hdf', output_filename), key=key, mode='w', format='table')
    return data

# Concatenate files
print("Concatenating small files...")
phenotype_info = concatenate_and_save(os.path.join(home, "ph_2", "hdf", "phenotype_info_*.hdf"), 'phenotype_info.hdf', 'phenotype_info')
min_cp_phenotype = concatenate_and_save(os.path.join(home, "ph_2", "hdf", "min_cp_phenotype_*.hdf"), 'min_cp_phenotype.hdf', 'min_cp_phenotype')

# Generate plots
print("Generating plots...")
plt.figure(figsize=(12, 6))
df_summary, _ = plot_count_heatmap(phenotype_info, shape='6W_sbs', return_summary=True)
save_plot(plt.gcf(), 'phenotype_count_heatmap.png')
df_summary.to_csv(os.path.join(qc_dir, 'phenotype_count_heatmap.csv'), index=False)

# Plot feature heatmaps for each cellular marker
features = ['cell_dapi_min', 'cell_cenpa_min', 'cell_coxiv_min', 'cell_wga_min']
for feature in features:
    plt.figure(figsize=(12, 6))
    df_summary, _ = plot_feature_heatmap(min_cp_phenotype, feature=feature, shape='6W_sbs', return_summary=True)
    save_plot(plt.gcf(), f'{feature}_heatmap.png')
    df_summary.to_csv(os.path.join(qc_dir, f'{feature}_heatmap.csv'), index=False)

num_rows = len(phenotype_info)
print(f"The number of cells extracted in the phenotype step is: {num_rows}")

print("QC analysis completed.")

print("Concatenating large cp files...")
cp_phenotype = concatenate_and_save(os.path.join(home, "ph_2", "hdf", "cp_phenotype_*.hdf"), 'cp_phenotype.hdf', 'cp_phenotype')
