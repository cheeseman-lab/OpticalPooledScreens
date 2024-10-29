# Import necessary modules and libraries
import snakemake
import ops.firesnake
from ops.firesnake import Snake
import ops.utils
import ops.triangle_hash as th
from joblib import Parallel, delayed
import pandas as pd

# Define plate layout
WELLS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']   # List of well identifiers
TILES_PH = list(range(1605))  # List of tile numbers + 1
TILES_SBS = list(range(377)) # List of tile numbers + 1

# Use cycle 1 for metadata read in
SBS_CYCLE = '1'

# Define determinant range, minimum score for initial sites for alignment
DET_RANGE = (0.06, 0.065)
SCORE = 0.1
INITIAL_SITES = [(5,0),(141,32),(370,86),(896,212),(1163,270),(1599,376)]

# Define the final output files for the pipeline
rule all:
    input:
        # Request individual files or list of files
        'merge_3/hdf/fast_alignment_all.hdf'

# Rule for fast alignment process
rule fast_alignment:
    input:
        # Define input files for fast alignment
        'metadata/20X_{{well}}.metadata.pkl'.format(),
        expand('process_ph/tables/20X_{{well}}_Tile-{tile}.phenotype_info.csv', tile=TILES_PH),
        'metadata/10X_c{cycle}-SBS-{cycle}_{{well}}.metadata.pkl'.format(cycle=SBS_CYCLE),
        expand('process_sbs/tables/10X_{{well}}_Tile-{tile}.sbs_info.csv', tile=TILES_SBS),
    output:
        # Define output files for fast alignment
        temp('merge_3/hdf/fast_alignment_{well}.hdf')
    run:
        # Assign input files
        f_ph_metadata = input[0]
        f_ph_info = input[1:len(TILES_PH) + 1]
        f_sbs_metadata = input[len(TILES_PH) + 1]
        f_sbs_info = input[(len(TILES_PH) + 2):]

        # Define function to read CSV files
        def get_file(f):
            try:
                return pd.read_csv(f)
            except pd.errors.EmptyDataError:
                pass

        # Read phenotype info CSV files in parallel
        arr_ph = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in f_ph_info)
        df_ph_info = pd.concat(arr_ph)

        # Ensure that i and j are not null, at least 4 cells per tile
        df_ph_info = df_ph_info[df_ph_info['i'].notnull() & df_ph_info['j'].notnull()]
        df_ph_info = df_ph_info.groupby(['well', 'tile']).filter(lambda x: len(x) > 3)

        # Read SBS info CSV files in parallel
        arr_sbs = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in f_sbs_info)
        df_sbs_info = pd.concat(arr_sbs)

        # Ensure that i and j are not null, at least 4 cells per tile
        df_sbs_info = df_sbs_info[df_sbs_info['i'].notnull() & df_sbs_info['j'].notnull()]
        df_sbs_info = df_sbs_info.groupby(['well', 'tile']).filter(lambda x: len(x) > 3)

        # Hash phenotype and SBS info, perform alignment
        df_ph_info_hash = (df_ph_info
            .pipe(ops.utils.gb_apply_parallel, ['tile'], th.find_triangles, n_jobs=threads, tqdm=False)
        )
        df_sbs_info_hash = (df_sbs_info
            .pipe(ops.utils.gb_apply_parallel, ['tile'], th.find_triangles, n_jobs=threads, tqdm=False)
            .rename(columns={'tile': 'site'})
        )

        # Read XY coordinates for phenotype and SBS
        df_ph_xy = (pd.read_pickle(f_ph_metadata)
            .rename(columns={'field_of_view': 'tile', 'x_data': 'x', 'y_data': 'y'})
            .set_index('tile')
            [['x', 'y']]
        )

        df_sbs_xy = (pd.read_pickle(f_sbs_metadata)
            .rename(columns={'field_of_view': 'tile', 'x_data': 'x', 'y_data': 'y'})
            .set_index('tile')
            [['x', 'y']]
        )

        # Perform multistep alignment
        df_align = th.multistep_alignment(
            df_ph_info_hash,
            df_sbs_info_hash,
            df_ph_xy,
            df_sbs_xy,
            det_range=DET_RANGE,
            score=SCORE,
            initial_sites=INITIAL_SITES,
            tqdn=False,
            n_jobs=threads
        )

        # Assign well identifier and write alignment results to HDF
        df_align.assign(well=wildcards.well).to_hdf(output[0], 'x', mode='w')

# Rule for combining alignment results from different wells
rule combine_alignment:
    input:
        expand('merge_3/hdf/fast_alignment_{well}.hdf', well=WELLS)
    output:
        'merge_3/hdf/fast_alignment_all.hdf'
    run:
        # Concatenate alignment results from different wells
        df_alignment = pd.concat([pd.read_hdf(f).assign(well=f[-6:-4]) for f in input])

        # Write combined alignment results to HDF
        df_alignment.to_hdf(output[0], 'x', mode='w')