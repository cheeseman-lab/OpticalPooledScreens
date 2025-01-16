#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ops.qc import *
from ops.firesnake import Snake

# Set screen directories
sbs_function_home = "/lab/barcheese01/screens"
sbs_function_dataset = "denali"
home = os.path.join(sbs_function_home, sbs_function_dataset)

# Create plots directory if it doesn't exist
qc_dir = os.path.join(home, 'sbs_1', 'qc')
os.makedirs(qc_dir, exist_ok=True)

def convert_column_to_string(df, column_name):
    df[column_name] = df[column_name].astype(str)
    return df

def read_hdf_generator(file_pattern):
    for file in glob(file_pattern):
        yield pd.read_hdf(file)

def save_plot(fig, filename):
    fig.savefig(os.path.join(qc_dir, filename))
    plt.close(fig)

def concatenate_and_save(file_pattern, output_filename, key):
    data = pd.concat(read_hdf_generator(file_pattern), ignore_index=True)
    
    # Convert problematic columns to string
    string_columns = ['gene_id_0', 'gene_id_1', 'gene_symbol_0', 'gene_symbol_1']
    for col in string_columns:
        if col in data.columns:
            data = convert_column_to_string(data, col)
    
    # Replace empty strings with np.nan
    data = data.replace(r'^\s*$', np.nan, regex=True)
    
    # Replace 'None' and 'nan' strings with np.nan
    data = data.replace(['None', 'nan'], np.nan)
    
    data.to_hdf(os.path.join(home, 'sbs_1', 'hdf', output_filename), key=key, mode='w', format='table')
    return data

# Read barcodes
df_design = pd.read_csv(os.path.join(home, 'sbs_1/pool10_design.csv'), index_col=None)
df_pool = df_design.query('dialout==[0,1]').drop_duplicates('sgRNA')
df_pool['prefix'] = df_pool.apply(lambda x: x.sgRNA[:x.prefix_length], axis=1) # 13
barcodes = df_pool['prefix']

# Concatenate files
print("Concatenating files...")
reads = concatenate_and_save(os.path.join(home, "sbs_1", "hdf", "reads_*.hdf"), 'reads.hdf', 'reads')
sbs_info = concatenate_and_save(os.path.join(home, "sbs_1", "hdf", "sbs_info_*.hdf"), 'sbs_info.hdf', 'sbs_info')
cells = concatenate_and_save(os.path.join(home, "sbs_1", "hdf", "cells_*.hdf"), 'cells.hdf', 'cells')

# Generate plots
print("Generating plots...")
plt.figure(figsize=(12, 6))
plot_mapping_vs_threshold(reads, barcodes, "peak")
save_plot(plt.gcf(), 'mapping_vs_threshold_peak.png')

plt.figure(figsize=(12, 6))
plot_mapping_vs_threshold(reads, barcodes, "Q_min")
save_plot(plt.gcf(), 'mapping_vs_threshold_qmin.png')

plt.figure(figsize=(12, 6))
plot_read_mapping_heatmap(reads, barcodes, shape='6W_sbs')
save_plot(plt.gcf(), 'read_mapping_heatmap.png')

plt.figure(figsize=(12, 6))
df_summary_one, _ = plot_cell_mapping_heatmap(cells, sbs_info, barcodes, mapping_to='one', mapping_strategy='gene_symbols', shape='6W_sbs', return_summary=True)
save_plot(plt.gcf(), 'cell_mapping_heatmap_one.png')
df_summary_one.to_csv(os.path.join(qc_dir, 'cell_mapping_heatmap_one.csv'), index=False)

plt.figure(figsize=(12, 6))
df_summary_any, _ = plot_cell_mapping_heatmap(cells, sbs_info, barcodes, mapping_to='any', mapping_strategy='gene_symbols', shape='6W_sbs', return_summary=True)
save_plot(plt.gcf(), 'cell_mapping_heatmap_any.png')
df_summary_any.to_csv(os.path.join(qc_dir, 'cell_mapping_heatmap_any.csv'), index=False)

plt.figure(figsize=(12, 6))
plot_reads_per_cell_histogram(cells, x_cutoff=30)
save_plot(plt.gcf(), 'reads_per_cell_histogram.png')

plt.figure(figsize=(12, 6))
plot_gene_symbol_histogram(cells, x_cutoff=3000)
save_plot(plt.gcf(), 'gene_symbol_histogram.png')

plt.figure(figsize=(12, 6))
barcodes_per_gene = plot_barcodes_per_gene(cells, cutoff=4)
save_plot(plt.gcf(), 'barcodes_per_gene_piechart.png')
barcodes_per_gene.to_csv(os.path.join(qc_dir, 'barcodes_per_gene.csv'), index=False)

num_rows = len(sbs_info)
print(f"The number of cells extracted in the sbs step is: {num_rows}")

# Calculate and print mapped single gene statistics
print("Calculating mapped single gene statistics...")
cells['mapped_single_gene'] = cells.apply(lambda x: True 
                    if (pd.notnull(x.gene_symbol_0) & pd.isnull(x.gene_symbol_1)) | (x.gene_symbol_0 == x.gene_symbol_1) 
                    else False, axis=1)

print(cells.mapped_single_gene.value_counts())

print("QC analysis completed.")