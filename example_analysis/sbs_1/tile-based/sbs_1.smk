# Import necessary modules and libraries
import pandas as pd
import os
from joblib import Parallel, delayed

import snakemake
import ops.firesnake
from ops.firesnake import Snake
from ops.imports import *
import ops.io

# Define lists of cycles
SBS_CYCLES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
CYCLE_FILES = None

# Define wells and tiles
WELLS = ['A1','A2','A3','B1','B2','B3'] 
TILES = list(range(377)) # List of tile numbers + 1

# Define channels
CHANNELS = None

# Define a pattern for preprocessing image files
preprocess_pattern = 'input_sbs_tif/10X_c{cycle}-SBS-{cycle}_{{well}}_Tile-{{tile}}.sbs.tif'

# Read and format guide data
df_design = pd.read_csv('sbs_1/pool10_design.csv', index_col=None)
df_pool = df_design.query('dialout==[0,1]').drop_duplicates('sgRNA')
df_pool['prefix'] = df_pool.apply(lambda x: x.sgRNA[:x.prefix_length], axis=1)
barcodes = df_pool['prefix']

# Define display ranges for different channels, recognized by ImageJ
DISPLAY_RANGES = [
    [500, 15000],  # Range for DAPI channel
    [100, 10000],  # Range for CY3 channel
    [100, 10000],  # Range for A594 channel
    [200, 25000],  # Range for CY5 channel
    [200, 25000]   # Range for CY7 channel
]

# Define LUTs (Lookup Tables) for different channels
LUTS = [
    ops.io.GRAY,    # Lookup table for DAPI channel
    ops.io.GREEN,   # Lookup table for CY3 channel
    ops.io.RED,     # Lookup table for A594 channel
    ops.io.MAGENTA, # Lookup table for CY5 channel
    ops.io.CYAN     # Lookup table for CY7 channel
]

# Define segmentation values
SEGMENTATION_CYCLE = -1
DAPI_INDEX = 0
CYTO_CHANNEL = 4
SEGMENT_METHOD = "cellpose"
NUCLEI_DIAMETER = 13.2
CELL_DIAMETER = 19.5
CYTO_MODEL = "cyto3"
THRESHOLD_DAPI = 1000  # Unused
THRESHOLD_CELL = 2500   # Unused
NUCLEUS_AREA = (20,800) # Unused

# Define base and read mapping values
THRESHOLD_READS = 315
BASES = 'GTAC' 
Q_MIN = 0

# Define function to read CSV files
def get_file(f):
    try:
        return pd.read_csv(f)
    except pd.errors.EmptyDataError:
        pass

# Defines the final output files for the pipeline, ensuring generation of files for each combination of well and tile
rule all:
    input:
        # Each file is generated for each combination of well and tile
        expand('process_sbs/tables/10X_{well}_Tile-{tile}.cells.csv', well=WELLS, tile=TILES),
        expand('process_sbs/tables/10X_{well}_Tile-{tile}.sbs_info.csv', well=WELLS, tile=TILES),
        expand('sbs_1/hdf/cells_{well}.hdf', well=WELLS),
        expand('sbs_1/hdf/reads_{well}.hdf', well=WELLS),
        expand('sbs_1/hdf/sbs_info_{well}.hdf', well=WELLS),
        
# Aligns images from each sequencing round 
rule align:
    priority: -1
    input:
        [preprocess_pattern.format(cycle=cycle) for cycle in SBS_CYCLES]
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif')
    run:
        data = [read(f) for f in input]
        print(len(data))
        Snake.align_SBS(
            output=output, 
            data=data, 
            method='SBS_mean', 
            cycle_files=CYCLE_FILES, 
            upsample_factor = 1, 
            n = 1, 
            keep_extras=False,
            display_ranges=DISPLAY_RANGES, 
            luts=LUTS
        )

# Applies Laplacian-of-Gaussian filter to all channels
rule transform_LoG:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.log.tif')
    run:
        Snake.transform_log(
            output=output, 
            data=input[0], 
            skip_index=0,
            display_ranges=DISPLAY_RANGES, 
            luts=LUTS
        )

# Computes standard deviation of SBS reads across cycles
rule compute_std:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.std.tif')
    run:
        Snake.compute_std(
            output=output, 
            data=input[0], 
            remove_index=0
        )

# Find local maxima of SBS reads across cycles
rule find_peaks:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.std.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif')
    run:
        Snake.find_peaks(
            output=output, 
            data=input[0]
        )

# Dilates sequencing channels to compensate for single-pixel alignment error.
rule max_filter:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.log.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.maxed.tif')
    run:
        Snake.max_filter(
            output=output,
            data=input[0],
            width=3,
            remove_index=0
        )

# Applies illumination correction to cycle 0
rule illumination_correction:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.aligned.tif',
        'illumination_correction/10X_c{cycle}-SBS-{cycle}_{{well}}.sbs.illumination_correction.tif'.format(cycle=SBS_CYCLES[SEGMENTATION_CYCLE]),
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.illumination_correction.tif')
    run:
        aligned = read(input[0])
        aligned_0 = aligned[0]
        print(aligned_0.shape)
        Snake.apply_illumination_correction(
            output=output, 
            data=aligned_0, 
            correction=input[1])

# Segments cells and nuclei using pre-defined methods
rule segment:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.illumination_correction.tif'
    output:
        temp('process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'),
        temp('process_sbs/images/10X_{well}_Tile-{tile}.cells.tif'),
    run:
        if SEGMENT_METHOD == "cellpose":
            Snake.segment_cellpose(
                output=output,
                data=input[0],
                dapi_index=DAPI_INDEX,
                cyto_index=CYTO_CHANNEL,
                nuclei_diameter=NUCLEI_DIAMETER,
                cell_diameter=CELL_DIAMETER,
                cyto_model=CYTO_MODEL
            )
        elif SEGMENT_METHOD == "cell_2019":
            Snake.segment_cell_2019(
                output=output,
                data=input[0],
                nuclei_threshold=THRESHOLD_DAPI,
                nuclei_area_min=NUCLEUS_AREA[0],
                nuclei_area_max=NUCLEUS_AREA[1],
                cell_threshold=THRESHOLD_CELL,
            )
        else:
            raise ValueError(f"Invalid SEGMENT_METHOD: {SEGMENT_METHOD}")

# Extract bases from peaks
rule extract_bases:
    input:
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.maxed.tif',
        'process_sbs/images/10X_{well}_Tile-{tile}.cells.tif',
    output:
        temp('process_sbs/tables/10X_{well}_Tile-{tile}.bases.csv')
    run:
        Snake.extract_bases(
            output=output, 
            peaks=input[0], 
            maxed=input[1], 
            cells=input[2], 
            threshold_peaks=THRESHOLD_READS, 
            bases=BASES, 
            wildcards=dict(wildcards)
        )

# Call reads
rule call_reads:
    input:
        'process_sbs/tables/10X_{well}_Tile-{tile}.bases.csv',
        'process_sbs/images/10X_{well}_Tile-{tile}.peaks.tif',
    output:
        temp('process_sbs/tables/10X_{well}_Tile-{tile}.reads.csv')
    run:
        Snake.call_reads(
            output=output, 
            df_bases=input[0], 
            peaks=input[1], 
        )

# Call cells
rule call_cells:
    input:
        'process_sbs/tables/10X_{well}_Tile-{tile}.reads.csv'
    output:
        temp('process_sbs/tables/10X_{well}_Tile-{tile}.cells.csv')
    run:
        Snake.call_cells(
            output=output, 
            df_reads=input[0], 
            df_pool=df_pool, 
            q_min=Q_MIN
        )

# Extract minimal phenotype features
rule sbs_cell_info:
    input: 
        'process_sbs/images/10X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process_sbs/tables/10X_{well}_Tile-{tile}.sbs_info.csv',
    run:
        Snake.extract_phenotype_minimal(
            output=output, 
            data_phenotype=input[0], 
            nuclei=input[0], 
            wildcards=wildcards
        )
        
# Rule for combining alignment results from different wells
rule merge_cells:
    input:
        expand('process_sbs/tables/10X_{{well}}_Tile-{tile}.cells.csv', tile=TILES),
    output:
        'sbs_1/hdf/cells_{well}.hdf',
    resources:
        mem_mb=96000  # Request 96 GB of memory
    run:
        arr_cells = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input)
        df_cells = pd.concat(arr_cells)
        df_cells.to_hdf(output[0], 'x', mode='w')
        
# Rule for combining alignment results from different wells
rule merge_reads:
    input:
        expand('process_sbs/tables/10X_{{well}}_Tile-{tile}.reads.csv', tile=TILES),
    output:
        'sbs_1/hdf/reads_{well}.hdf',
    resources:
        mem_mb=96000  # Request 96 GB of memory
    run:
        arr_reads = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input)
        df_reads = pd.concat(arr_reads)
        df_reads.to_hdf(output[0], 'x', mode='w')
        
# Rule for combining alignment results from different wells
rule merge_sbs_info:
    input:
        expand('process_sbs/tables/10X_{{well}}_Tile-{tile}.sbs_info.csv', tile=TILES),
    output:
        'sbs_1/hdf/sbs_info_{well}.hdf',
    resources:
        mem_mb=96000  # Request 96 GB of memory
    run:
        arr_sbs_info = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input)
        df_sbs_info = pd.concat(arr_sbs_info)
        df_sbs_info.to_hdf(output[0], 'x', mode='w')