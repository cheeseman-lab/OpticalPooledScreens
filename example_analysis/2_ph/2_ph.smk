# Import necessary modules and libraries
import os
import pandas as pd
from joblib import Parallel, delayed

import snakemake
import ops.firesnake
from ops.firesnake import Snake
from ops.imports import *
import ops.io

# Define wells and tiles
WELLS = ["A1", "A2"]
TILES = [1, 100]

# Define channels
CHANNELS = None

# Define a pattern for preprocessing image files
preprocess_pattern = "input_ph_tif/20X_{well}_Tile-{tile}.phenotype.tif"

# Define segmentation values
SEGMENT_METHOD = "cellpose"
DAPI_INDEX = 0
CYTO_CHANNEL = 1
NUCLEI_DIAMETER = 47.1
CELL_DIAMETER = 55.3
CYTO_MODEL = "cyto3"
BACKGROUND_CHANNEL = 3  # Unused
BACKGROUND_THRESHOLD_OFFSET = 80  # Unused
BACKGROUND_QUANTILE = {"low": 0.05, "mid": 0.025, "high": 0.01}  # Unused
CELL_COUNT_THRESHOLDS = (1400, 2500)  # Unused
BACKGROUND_EROSION = 50  # Unused
BACKGROUND_SMOOTH = 3  # Unused
THRESHOLD_DAPI = 2500  # Unused
THRESHOLD_CELL = 500  # Unused
NUCLEUS_AREA = (250, 3500)  # Unused

# Define cellprofiler parameters
FOCI_CHANNEL = 1
CHANNEL_NAMES = ["dapi", "cenpa", "coxiv", "wga"]


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
        expand(
            "process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv",
            well=WELLS,
            tile=TILES,
        ),
        expand(
            "process_ph/tables/20X_{well}_Tile-{tile}.cp_phenotype.csv",
            well=WELLS,
            tile=TILES,
        ),
        expand("ph_2/hdf/phenotype_info_{well}.hdf", well=WELLS),
        expand("ph_2/hdf/min_cp_phenotype_{well}.hdf", well=WELLS),
        expand("ph_2/hdf/cp_phenotype_{well}.hdf", well=WELLS),


# Applies illumination correction
rule apply_illumination_correction:
    input:
        preprocess_pattern,
        "illumination_correction/20X_{well}.phenotype.illumination_correction.tif",
    output:
        temp("process_ph/images/20X_{well}_Tile-{tile}.corrected.tif"),
    run:
        print(input[0])
        print(input[1])
        Snake.apply_illumination_correction(
            output=output, data=input[0], correction=input[1]
        )


# Segments cells and nuclei using pre-defined methods
rule segment:
    input:
        "process_ph/images/20X_{well}_Tile-{tile}.corrected.tif",
    output:
        "process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif",
        "process_ph/images/20X_{well}_Tile-{tile}.cells.tif",
    resources:
        mem_mb=16000,
    run:
        if SEGMENT_METHOD == "cellpose":
            Snake.segment_cellpose(
                output=output,
                data=input[0],
                dapi_index=DAPI_INDEX,
                cyto_index=CYTO_CHANNEL,
                nuclei_diameter=NUCLEI_DIAMETER,
                cell_diameter=CELL_DIAMETER,
                cyto_model=CYTO_MODEL,
            )
        elif SEGMENT_METHOD == "cell_2022":
            Snake.segment_cell_2022(
                output=output,
                data=input[0],
                nuclei_threshold=THRESHOLD_DAPI,
                nuclei_area_min=NUCLEUS_AREA[0],
                nuclei_area_max=NUCLEUS_AREA[1],
                channel=BACKGROUND_CHANNEL,
                background_offset=BACKGROUND_THRESHOLD_OFFSET,
                cell_count_thresholds=CELL_COUNT_THRESHOLDS,
                background_quantile=BACKGROUND_QUANTILE,
                smooth=BACKGROUND_SMOOTH,
                erosion=BACKGROUND_EROSION,
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
            raise ValueError(
                f"Invalid SEGMENT_METHOD: {SEGMENT_METHOD}. Choose 'cellpose', 'cell_2022', or 'cell_2019'."
            )


# Rule to extract cytoplasmic masks from segmented nuclei, cells
rule identify_cytoplasm:
    input:
        "process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif",
        "process_ph/images/20X_{well}_Tile-{tile}.cells.tif",
    output:
        temp("process_ph/images/20X_{well}_Tile-{tile}.cytoplasms.tif"),
    run:
        if SEGMENT_METHOD == "cellpose":
            Snake.identify_cytoplasm_cellpose(
                output=output, nuclei=input[0], cells=input[1]
            )
        elif SEGMENT_METHOD in ["cell_2019", "cell_2022"]:
            Snake.identify_cytoplasm(output=output, nuclei=input[0], cells=input[1])
        else:
            raise ValueError(
                f"Invalid SEGMENT_METHOD: {SEGMENT_METHOD}. Choose 'cellpose', 'cell_2022', or 'cell_2019'."
            )


# Rule to extract minimal phenotype information from segmented nuclei images
rule phenotype_info:
    input:
        "process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif",
    output:
        "process_ph/tables/20X_{well}_Tile-{tile}.phenotype_info.csv",
    run:
        Snake.extract_phenotype_minimal(
            output=output, data_phenotype=input[0], nuclei=input[0], wildcards=wildcards
        )


# Rule for combining phenotype info results from different wells
rule merge_ph_info:
    input:
        expand(
            "process_ph/tables/20X_{{well}}_Tile-{tile}.phenotype_info.csv", tile=TILES
        ),
    output:
        "ph_2/hdf/phenotype_info_{well}.hdf",
    resources:
        mem_mb=96000,
    run:
        arr_ph_info = Parallel(n_jobs=threads)(
            delayed(get_file)(file) for file in input
        )
        df_ph_info = pd.concat(arr_ph_info)
        df_ph_info.to_hdf(output[0], "x", mode="w")


# Rule to extract full phenotype information using CellProfiler from phenotype images
rule extract_phenotype_cp:
    input:
        "process_ph/images/20X_{well}_Tile-{tile}.corrected.tif",
        "process_ph/images/20X_{well}_Tile-{tile}.nuclei.tif",
        "process_ph/images/20X_{well}_Tile-{tile}.cells.tif",
        "process_ph/images/20X_{well}_Tile-{tile}.cytoplasms.tif",
    output:
        "process_ph/tables/20X_{well}_Tile-{tile}.cp_phenotype.csv",
    benchmark:
        "process_ph/benchmark/20X_{well}_Tile-{tile}.benchmark_cp_phenotype.tsv"
    run:
        Snake.extract_phenotype_cp_multichannel(
            output=output,
            data_phenotype=input[0],
            nuclei=input[1],
            cells=input[2],
            wildcards=wildcards,
            cytoplasms=input[3],
            foci_channel=FOCI_CHANNEL,
            channel_names=CHANNEL_NAMES,
        )


# Rule for combining phenotype results from different wells
rule merge_ph_cp:
    input:
        expand(
            "process_ph/tables/20X_{{well}}_Tile-{tile}.cp_phenotype.csv", tile=TILES
        ),
    output:
        "ph_2/hdf/cp_phenotype_{well}.hdf",
        "ph_2/hdf/min_cp_phenotype_{well}.hdf",
    resources:
        mem_mb=96000,
    run:
        arr_ph_cp = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input)
        df_ph_cp = pd.concat(arr_ph_cp)
        df_ph_cp.to_hdf(output[0], "x", mode="w")

        df_min_ph_cp = df_ph_cp[
            [
                "well",
                "tile",
                "label",
                "cell_i",
                "cell_j",
                "cell_bounds_0",
                "cell_bounds_1",
                "cell_bounds_2",
                "cell_bounds_3",
                "cell_dapi_min",
                "cell_cenpa_min",
                "cell_coxiv_min",
                "cell_wga_min",
            ]
        ]
        df_min_ph_cp.to_hdf(output[1], "x", mode="w")
