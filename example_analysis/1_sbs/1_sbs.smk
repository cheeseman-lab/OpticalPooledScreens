# Import necessary modules and libraries
import os

import pandas as pd
from joblib import Parallel, delayed
import snakemake

from ops.sbs_smk import Snake_sbs
from ops.imports import read
import ops.io

# Output directory for notebook results
INPUT_FILES_DIR = "input"
OUTPUT_FILES_DIR = "output"
PROCESSING_FILES_DIR = "processing"

# Define lists of cycles
SBS_CYCLES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
CYCLE_FILES = None

# Define wells and tiles
WELLS = ['A1']
TILES = [50]

# Define channels
CHANNELS = None

# Define the file pattern for preprocessing image files
PREPROCESS_PATTERN = "{input_dir}/10X_c{cycle}-SBS-{cycle}_{{well}}_Tile-{{tile}}.sbs.tif"

# Define display ranges for different channels, recognized by ImageJ
DISPLAY_RANGES = [
    [500, 15000],  # Range for DAPI channel
    [100, 10000],  # Range for CY3 channel
    [100, 10000],  # Range for A594 channel
    [200, 25000],  # Range for CY5 channel
    [200, 25000],  # Range for CY7 channel
]

# Define LUTs (Lookup Tables) for different channels
LUTS = [
    ops.io.GRAY,  # Lookup table for DAPI channel
    ops.io.GREEN,  # Lookup table for CY3 channel
    ops.io.RED,  # Lookup table for A594 channel
    ops.io.MAGENTA,  # Lookup table for CY5 channel
    ops.io.CYAN,  # Lookup table for CY7 channel
]

# Define cycle to use for segmentation, -1 for last cycle
SEGMENTATION_CYCLE = -1

# Define illumination correction file for use in segmentation
ICF_FILE_NAME = "10X_c{cycle}-SBS-{cycle}_{{well}}.sbs.illumination_correction.tif"

# Define Cellpose segmentation parameters
DAPI_INDEX = 0
CYTO_CHANNEL = 4

# Parameters for cellpose method
NUCLEI_DIAMETER = 13.2  # Calibrate with CellPose
CELL_DIAMETER = 19.5  # Calibrate with CellPose
CYTO_MODEL = "cyto3"

# Define parameters for extracting bases
DF_DESIGN_PATH = "input/pool10_design.csv"
THRESHOLD_READS = 315
BASES = "GTAC"

# Define parameters for read mapping
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
        expand(f'{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.reads.csv', well=WELLS, tile=TILES),
        expand(f'{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.cells.csv', well=WELLS, tile=TILES),
        expand(f'{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.sbs_info.csv', well=WELLS, tile=TILES),
        
# Aligns images from each sequencing round 
rule align:
    input:
        [PREPROCESS_PATTERN.format(input_dir=INPUT_FILES_DIR, cycle=cycle) for cycle in SBS_CYCLES]
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.aligned.tif")
    run:
        # Read each cycle image into a list
        data = [read(f) for f in input]
        
        # Print number of data points for verification
        print(f"Number of images loaded: {len(data)}")

        # Call the alignment function from Snake_sbs
        Snake_sbs.align_SBS(
            output=output, 
            data=data, 
            method='SBS_mean', 
            cycle_files=CYCLE_FILES, 
            upsample_factor=1, 
            n=1, 
            keep_extras=False,
            display_ranges=DISPLAY_RANGES, 
            luts=LUTS
        )

# Applies Laplacian-of-Gaussian filter to all channels
rule transform_LoG:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.aligned.tif"
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.log.tif")
    run:
        Snake_sbs.transform_log(
            data=input[0], 
            output=output, 
            skip_index=0,
            display_ranges=DISPLAY_RANGES, 
            luts=LUTS
        )

# Computes standard deviation of SBS reads across cycles
rule compute_std:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.log.tif"
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.std.tif")
    run:
        Snake_sbs.compute_std(
            data=input[0], 
            output=output, 
            remove_index=0
        )

# Find local maxima of SBS reads across cycles
rule find_peaks:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.std.tif"
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.peaks.tif")
    run:
        Snake_sbs.find_peaks(
            data=input[0],
            output=output
        )

# Dilates sequencing channels to compensate for single-pixel alignment error.
rule max_filter:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.log.tif"
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.maxed.tif")
    run:
        Snake_sbs.max_filter(
            data=input[0],
            output=output,
            width=3,
            remove_index=0
        )

# Applies illumination correction to cycle 0
rule illumination_correction:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.aligned.tif",
        'input/10X_c{cycle}-SBS-{cycle}_{{well}}.sbs.illumination_correction.tif'.format(cycle=SBS_CYCLES[SEGMENTATION_CYCLE]),
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.illumination_correction.tif")
    run:
        aligned = read(input[0])
        aligned_0 = aligned[0]
        print(aligned_0.shape)
        Snake_sbs.apply_illumination_correction(
            data=aligned_0, 
            correction=input[1],
            output=output, 
        )

# Segments cells and nuclei using pre-defined methods
rule segment:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.illumination_correction.tif",
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.nuclei.tif"),
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.cells.tif")
    run:
        Snake_sbs.segment_cellpose(
            data=input[0],
            output=output,
            dapi_index=DAPI_INDEX,
            cyto_index=CYTO_CHANNEL,
            nuclei_diameter=NUCLEI_DIAMETER,
            cell_diameter=CELL_DIAMETER,
            cyto_model=CYTO_MODEL
        )

# Extract bases from peaks
rule extract_bases:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.peaks.tif",
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.maxed.tif",
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.cells.tif",
    output:
        temp(f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.bases.csv")
    run:
        Snake_sbs.extract_bases(
            peaks=input[0], 
            maxed=input[1], 
            cells=input[2], 
            output=output, 
            threshold_peaks=THRESHOLD_READS, 
            bases=BASES, 
            wildcards=dict(wildcards)
        )

# Call reads
rule call_reads:
    input:
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.bases.csv",
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.peaks.tif",
    output:
        f"{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.reads.csv",
    run:
        Snake_sbs.call_reads(
            df_bases=input[0], 
            peaks=input[1], 
            output=output, 
        )

# Call cells
rule call_cells:
    input:
        f"{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.reads.csv",
    output:
        f"{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.cells.csv",
    run:
        df_design = pd.read_csv(DF_DESIGN_PATH)
        df_pool = df_design.query("dialout==[0,1]").drop_duplicates("sgRNA")

        Snake_sbs.call_cells(
            df_reads=input[0], 
            output=output, 
            df_pool=df_pool, 
            q_min=Q_MIN
        )

# Extract minimal phenotype features
rule sbs_cell_info:
    input: 
        f"{PROCESSING_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.nuclei.tif",
    output:
        f"{OUTPUT_FILES_DIR}/10X_{{well}}_Tile-{{tile}}.sbs_info.csv",
    run:
        Snake_sbs.extract_phenotype_minimal(
            data_phenotype=input[0], 
            nuclei=input[0], 
            output=output, 
            wildcards=wildcards
        )