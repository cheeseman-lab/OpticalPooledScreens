# Import necessary modules and libraries
import pandas as pd
from joblib import Parallel, delayed

import snakemake
import ops.firesnake
from ops.firesnake import Snake

# Define determinant range, minimum score, threshold for all alignments
det_range = (0.06, 0.065)
score = 0.1
threshold = 2

# Read data from HDF file
df_align_no_gate = pd.read_hdf('merge_3/hdf/fast_alignment_all.hdf')

# Filter DataFrame based on the filtering condition
df_align = df_align_no_gate[
    (df_align_no_gate["determinant"] >= det_range[0]) &
    (df_align_no_gate["determinant"] <= det_range[1]) &
    (df_align_no_gate["score"] > score)
]

# Convert numeric columns to integers before extracting
df_align['tile'] = df_align['tile'].astype(int)
df_align['site'] = df_align['site'].astype(int)

# Extract values of 'well', 'tile', and 'site' columns from aligned data and transpose them
WELLS, TILES_PH, SITES_SBS = df_align[['well', 'tile', 'site']].values.T

# Define function to read CSV files
def get_file(f):
    try:
        return pd.read_csv(f)
    except pd.errors.EmptyDataError:
        pass

# Define the final output files for the pipeline
rule all:
    input:
        # request individual files or list of files
        expand('alignment/{well}_Tile-{tile}_Site-{site}.merge.csv', zip, well=WELLS, tile=TILES_PH, site=SITES_SBS),
        'merge_3/hdf/merge_all.hdf',
    
rule merge:
    input:
        'process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv',
        'process_sbs/tables/10X_{well}_Tile-{site}.sbs_info.csv'
    output:
        temp('alignment/{well}_Tile-{tile}_Site-{site}.merge.csv')
    run:
        well_mask = df_align['well'] == wildcards.well
        tile_mask = df_align['tile'] == int(wildcards.tile)
        site_mask = df_align['site'] == int(wildcards.site)
        
        combined_mask = well_mask & tile_mask & site_mask
        
        alignment_vec = df_align[combined_mask].iloc[0]

        Snake.merge_triangle_hash(output=output, df_0=input[0], df_1=input[1], alignment=alignment_vec, threshold=threshold)

# Rule for combining alignment results
rule combine_merge:
    input:
        expand('alignment/{well}_Tile-{tile}_Site-{site}.merge.csv',
               zip, well=WELLS, tile=TILES_PH, site=SITES_SBS)
    output:
        'merge_3/hdf/merge_all.hdf',
    resources:
        mem_mb=96000
    run:
        arr_merge = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input)
        df_merge = pd.concat(arr_merge)
        df_merge.to_hdf(output[0], 'x', mode='w')