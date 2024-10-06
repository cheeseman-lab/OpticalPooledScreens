import ops
import os
import glob
import tifffile
import ops.filenames
from ops.preprocessing_smk import *
from ops.process import calculate_illumination_correction
from ops.io import save_stack as save

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
            "metadata/10X_c{cycle}-SBS-{cycle}_{well}.metadata.pkl",
            well=WELLS,
            cycle=SBS_CYCLES,
        ),
        expand("metadata/20X_{well}.metadata.pkl", well=WELLS),
        expand(
            "input_sbs_tif/10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.tif",
            well=WELLS,
            cycle=SBS_CYCLES,
            tile=SBS_TILES,
        ),
        expand(
            "input_ph_tif/20X_{well}_Tile-{tile}.phenotype.tif",
            well=WELLS,
            tile=PH_TILES,
        ),
        expand(
            "illumination_correction/10X_c{cycle}-SBS-{cycle}_{well}.sbs.illumination_correction.tif",
            cycle=SBS_CYCLES,
            well=WELLS,
        ),
        expand(
            "illumination_correction/20X_{well}.phenotype.illumination_correction.tif",
            well=WELLS,
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
        "metadata/10X_c{cycle}-SBS-{cycle}_{well}.metadata.pkl",
    resources:
        mem_mb=96000,
    run:
        os.makedirs("metadata", exist_ok=True)
        metadata = Snake_preprocessing.extract_metadata_tile(
            output=output[0],
            files=input,
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset,
            parse_function_tiles=True,
        )


# Extract metadata for PH images
rule extract_metadata_ph:
    input:
        lambda wildcards: glob.glob(
            PH_INPUT_PATTERN_METADATA.format(well=wildcards.well)
        ),
    output:
        "metadata/20X_{well}.metadata.pkl",
    resources:
        mem_mb=96000,
    run:
        os.makedirs("metadata", exist_ok=True)
        metadata = Snake_preprocessing.extract_metadata_tile(
            output=output[0],
            files=input,
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset,
            parse_function_tiles=True,
        )


# Convert SBS ND2 files to TIFF
rule convert_sbs:
    input:
        lambda wildcards: glob.glob(
            SBS_INPUT_PATTERN.format(
                cycle=wildcards.cycle, well=wildcards.well
            ).format(tile=f"{int(wildcards.tile):03d}")
        ),
    output:
        "input_sbs_tif/10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.tif",
    run:
        os.makedirs("input_sbs_tif", exist_ok=True)
        image_array, fov_description = Snake_preprocessing.convert_to_tif_tile(
            file=input[0],
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset,
            channel_order_flip=True,
        )
        output_filename = ops.filenames.name_file(fov_description)
        print(output_filename)
        save(output_filename, image_array)


# Convert PH ND2 files to TIFF
rule convert_ph:
    input:
        lambda wildcards: glob.glob(
            PH_INPUT_PATTERN.format(well=wildcards.well).format(
                tile=f"{int(wildcards.tile):03d}"
            )
        ),
    output:
        "input_ph_tif/20X_{well}_Tile-{tile}.phenotype.tif",
    run:
        os.makedirs("input_ph_tif", exist_ok=True)

        image_array, fov_description = Snake_preprocessing.convert_to_tif_tile(
            file=input[0],
            parse_function_home=parse_function_home,
            parse_function_dataset=parse_function_dataset,
            channel_order_flip=True,
        )
        output_filename = ops.filenames.name_file(fov_description)
        print(output_filename)
        save(output_filename, image_array)


# Calculate illumination correction for sbs files
rule calculate_icf_sbs:
    input:
        lambda wildcards: expand(
            "input_sbs_tif/10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.tif",
            cycle=wildcards.cycle,
            well=wildcards.well,
            tile=SBS_TILES,
        ),
    output:
        "illumination_correction/10X_c{cycle}-SBS-{cycle}_{well}.sbs.illumination_correction.tif",
    resources:
        mem_mb=96000,
    run:
        os.makedirs("illumination_correction", exist_ok=True)
        input_files = list(input)
        icf = calculate_illumination_correction(input_files, threading=-3)
        save(output[0], icf)


# Calculate illumination correction for ph files
rule calculate_icf_ph:
    input:
        lambda wildcards: expand(
            "input_ph_tif/20X_{well}_Tile-{tile}.phenotype.Channel-{{channel}}.tif",
            well=wildcards.well,
            channel=wildcards.channel,
            tile=PH_TILES,
        ),
    output:
        "illumination_correction/20X_{well}.phenotype.Channel-{channel}.illumination_correction.tif",
    resources:
        mem_mb=96000,
    run:
        os.makedirs("illumination_correction", exist_ok=True)
        input_files = list(input)
        icf = calculate_illumination_correction(input_files, threading=-3)
        save(output[0], icf)
