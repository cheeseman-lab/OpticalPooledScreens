# Import necessary modules and libraries
import snakemake
import ops.firesnake
from ops.firesnake import Snake
import pandas as pd

# Define determinant range, minimum score for all alignments
det_range = (0.06, 0.065)
score = 0.1

# Construct a string representing the filtering condition
gate = '{0} <= determinant <= {1} & score > {2}'.format(*det_range, score)

# Read data from HDF file
df_align_no_gate = pd.read_hdf('merge_3/fast_alignment_all.hdf')

# Filter DataFrame based on the filtering condition
df_align = df_align_no_gate.query(gate)

# Calculate the number of cells filtered out after applying the gate
final_filtered_out = len(df_align_no_gate) - len(df_align)
print("Number of cells filtered out after applying the gate:", final_filtered_out)

# Extract values of 'well', 'tile', and 'site' columns from aligned data and transpose them
WELLS, TILES_PH, SITES_SBS = df_align[['well', 'tile', 'site']].values.T

# Define the final output files for the pipeline
rule all:
    input:
        # request individual files or list of files
        expand('merge_3/hdf/merge_{well}.hdf', well=WELLS),
#        expand('alignment/{well}_Tile-{tile}_Site-{site}.merge.csv', zip, well=WELLS, tile=TILES_PH, site=SITES_SBS),

    
# Rule for merging ph and sbs based on alignment
rule merge:
    input:
        'process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv',
        'process_sbs/tables/10X_{well}_Tile-{site}.sbs_info.csv'
    output:
        'alignment/{well}_Tile-{tile}_Site-{site}.merge.csv'
    run:
        # Create boolean masks for each condition
        well_mask = df_align['well'] == wildcards.well
        tile_mask = df_align['tile'] == int(wildcards.tile)
        site_mask = df_align['site'] == int(wildcards.site)
        
        # Combine the masks with logical AND (&) operator
        combined_mask = well_mask & tile_mask & site_mask
        
        # Use the combined mask to filter the DataFrame
        alignment_vec = df_align[combined_mask].iloc[0]
        
        # Call Snake.merge_triangle_hash() with the filtered alignment vector
        Snake.merge_triangle_hash(output=output, df_0=input[0], df_1=input[1], alignment=alignment_vec)

# Rule for combining alignment results from different wells
rule merge_merge:
    input:
        expand('alignment/{{well}}_Tile-{tile}_Site-{site}.merge.csv', tile=TILES_PH, site=SITES_SBS),
    output:
        'merge_3/hdf/merge_{well}.hdf',
    run:
        # Concatenate alignment results from different wells
        arr_merge = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input)
        df_merge = pd.concat(arr_merge)
        
        # Write combined alignment results to HDF
        df_merge.to_hdf(output[0], 'x', mode='w')