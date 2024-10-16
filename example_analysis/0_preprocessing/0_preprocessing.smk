import ops
import os
import glob
import tifffile
import ops.filenames
from ops.preprocessing_smk import *
from ops.process import calculate_illumination_correction
from ops.io import save_stack as save

# Directory for loading and saving files
METADATA_DIR = "output/metadata"
SBS_TIF_DIR = "output/sbs_tif"
PH_TIF_DIR = "output/ph_tif"
IC_DIR = "output/illumination_correction"

# Define wells to preprocess
WELLS = ["A1", "A2"]

# Define SBS cycles/tiles to preprocess
SBS_CYCLES = list(range(1, 12))
SBS_TILES = [1, 100]

# Define phenotype tiles to preprocess
PH_TILES = [1, 100]

# Parse function parameters
PARSE_FUNCTION_HOME = "input"
PARSE_FUNCTION_DATASET = "example_dataset"

# File patterns for SBS and PH images with placeholders (find all tiles to compile metadata)
SBS_INPUT_PATTERN_METADATA = "input/sbs/*C{cycle}_Wells-{well}_Points-*__Channel*.nd2"
PH_INPUT_PATTERN_METADATA = "input/ph/*Wells-{well}_Points-*__Channel*.nd2"

# File patterns for SBS and PH images
SBS_INPUT_PATTERN = "input/sbs/*C{cycle}_Wells-{well}_Points-{tile:0>3}__Channel*.nd2"
PH_INPUT_PATTERN = "input/ph/*Wells-{well}_Points-{tile:0>3}__Channel*.nd2"
# phenotpye example files too large to be included on GitHub


# Final output files
rule all:
    input:
        expand(
            f"{METADATA_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}.metadata.pkl",
            well=WELLS,
            cycle=SBS_CYCLES,
        ),
        expand(
            f"{METADATA_DIR}/20X_{{well}}.metadata.pkl",
            well=WELLS,
        ),
        expand(
            f"{SBS_TIF_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}_Tile-{{tile}}.sbs.tif",
            well=WELLS,
            cycle=SBS_CYCLES,
            tile=SBS_TILES,
        ),
        expand(
            f"{PH_TIF_DIR}/20X_{{well}}_Tile-{{tile}}.phenotype.tif",
            well=WELLS,
            tile=PH_TILES,
        ),
        expand(
            f"{IC_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}.sbs.illumination_correction.tif",
            well=WELLS,
            cycle=SBS_CYCLES,
            tile=SBS_TILES,
        ),
        expand(
            f"{IC_DIR}/20X_{{well}}.phenotype.illumination_correction.tif",
            well=WELLS,
            tile=PH_TILES,
        ),


# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        lambda wildcards: glob.glob(
            SBS_INPUT_PATTERN_METADATA.format(
                cycle=wildcards.cycle, well=wildcards.well
            )
        ),
    output:
        f"{METADATA_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}.metadata.pkl",
    run:
        os.makedirs("metadata", exist_ok=True)
        metadata = Snake_preprocessing.extract_metadata_tile(
            files=input,
            parse_function_home=PARSE_FUNCTION_HOME,
            parse_function_dataset=PARSE_FUNCTION_DATASET,
            parse_function_tiles=True,
            output=output,
        )


# Extract metadata for PH images
rule extract_metadata_ph:
    input:
        lambda wildcards: glob.glob(
            PH_INPUT_PATTERN_METADATA.format(well=wildcards.well)
        ),
    output:
        f"{METADATA_DIR}/20X_{{well}}.metadata.pkl",
    run:
        metadata = Snake_preprocessing.extract_metadata_tile(
            files=input,
            parse_function_home=PARSE_FUNCTION_HOME,
            parse_function_dataset=PARSE_FUNCTION_DATASET,
            parse_function_tiles=True,
            output=output,
        )


# Convert SBS ND2 files to TIFF
rule convert_sbs:
    input:
        lambda wildcards: glob.glob(
            SBS_INPUT_PATTERN.format(
                cycle=wildcards.cycle,
                well=wildcards.well,
                tile=f"{int(wildcards.tile):03d}",
            )
        ),
    output:
        f"{SBS_TIF_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}_Tile-{{tile}}.sbs.tif",
    run:
        image_array, fov_description = Snake_preprocessing.convert_to_tif_tile(
            file=input[0],
            parse_function_home=PARSE_FUNCTION_HOME,
            parse_function_dataset=PARSE_FUNCTION_DATASET,
            channel_order_flip=True,
        )
        save(output[0], image_array)


# Convert PH ND2 files to TIFF
rule convert_ph:
    input:
        lambda wildcards: glob.glob(
            PH_INPUT_PATTERN.format(
                well=wildcards.well,
                tile=f"{int(wildcards.tile):03d}",
            )
        ),
    output:
        f"{PH_TIF_DIR}/20X_{{well}}_Tile-{{tile}}.phenotype.tif",
    run:
        image_array, fov_description = Snake_preprocessing.convert_to_tif_tile(
            file=input[0],
            parse_function_home=PARSE_FUNCTION_HOME,
            parse_function_dataset=PARSE_FUNCTION_DATASET,
            channel_order_flip=True,
        )
        save(output[0], image_array)


# Calculate illumination correction for sbs files
rule calculate_icf_sbs:
    input:
        lambda wildcards: expand(
            f"{SBS_TIF_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}_Tile-{{tile}}.sbs.tif",
            cycle=wildcards.cycle,
            well=wildcards.well,
            tile=SBS_TILES,
        ),
    output:
        f"{IC_DIR}/10X_c{{cycle}}-SBS-{{cycle}}_{{well}}.sbs.illumination_correction.tif",
    run:
        input_files = list(input)
        icf = calculate_illumination_correction(input_files, threading=-3)
        save(output[0], icf)


# Calculate illumination correction for ph files
rule calculate_icf_ph:
    input:
        lambda wildcards: expand(
            f"{PH_TIF_DIR}/20X_{{well}}_Tile-{{tile}}.phenotype.tif",
            well=wildcards.well,
            tile=PH_TILES,
        ),
    output:
        f"{IC_DIR}/20X_{{well}}.phenotype.illumination_correction.tif",
    run:
        os.makedirs("illumination_correction", exist_ok=True)
        input_files = list(input)
        icf = calculate_illumination_correction(input_files, threading=-3)
        save(output[0], icf)
