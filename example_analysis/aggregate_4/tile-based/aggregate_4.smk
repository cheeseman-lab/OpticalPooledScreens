# Import necessary modules
import os
import pandas as pd
import numpy as np
from ops.imports import *
from ops.aggregate import *

# Feature settings
POPULATION_FEATURE = 'gene_symbol_0'
FEATURE_START = 'nucleus_dapi_mean'

# Control settings
CONTROL_PREFIX = 'nontargeting'
GROUP_COLUMNS = ['well']
INDEX_COLUMNS = ['tile', 'cell_0']
CAT_COLUMNS = ['gene_symbol_0', 'sgRNA_0']

# Threshold settings
THRESHOLDS = {
    'nucleus_dapi_median': (7, 'greater'),
}

# Channel settings
CHANNEL_DICT = None
PLOTTING_DICT = {
    'dapi': {'filename': 'filename', 'channel': 0},
    'coxiv': {'filename': 'filename', 'channel': 1},
    'cenpa': {'filename': 'filename', 'channel': 2},
    'wga': {'filename': 'filename', 'channel': 3}
}
CHANNELS = list(PLOTTING_DICT.keys())

# Load gene-sgRNA combinations
df_design = pd.read_csv('aggregate_4/pool10_design.csv', index_col=None)
df_pool = df_design.query('dialout==[0,1]').drop_duplicates('sgRNA')
df_barcodes = df_pool[['gene_id', 'sgRNA', 'gene_symbol']]
df_barcodes.columns = ['gene_id', 'sgRNA_0', 'gene_symbol_0']
df_barcodes = (df_barcodes[['gene_symbol_0', 'sgRNA_0']]
               .drop_duplicates()
               .pipe(explode_with_plotting_dict, PLOTTING_DICT))

DISPLAY_RANGES = {
    'dapi': [(0, 14000)],
    'coxiv': [(0, 13000)],
    'cenpa': [(0, 6000)],
    'wga': [(350, 2000)]
}

# Final target rule
rule all:
    input:
        'aggregate_4/hdf/standardized_data.hdf',
        'aggregate_4/hdf/transformed_data.hdf',
        'aggregate_4/hdf/mitotic_montage_data.hdf',
        'aggregate_4/hdf/interphase_montage_data.hdf',
        'aggregate_4/csv/mitotic_gene_data.csv',
        'aggregate_4/csv/interphase_gene_data.csv',
        'aggregate_4/csv/all_gene_data.csv',
        expand('montage/mitotic/{gene}_{sgrna}_{channel}.tif',
               zip, 
               gene=df_barcodes['gene_symbol_0'], 
               sgrna=df_barcodes['sgRNA_0'],
               channel=df_barcodes['channel']),
        expand('montage/interphase/{gene}_{sgrna}_{channel}.tif',
               zip, 
               gene=df_barcodes['gene_symbol_0'], 
               sgrna=df_barcodes['sgRNA_0'],
               channel=df_barcodes['channel']),        

# Clean and transform data
rule clean_and_transform:
    input:
        'merge_3/hdf/merged_final.hdf',
        'aggregate_4/transformations.csv'
    output:
        'aggregate_4/hdf/transformed_data.hdf'
    resources:
        mem_mb=500000
    priority: 1
    run:
        raw_df = pd.read_hdf(input[0])
        clean_df = clean_cell_data(
            raw_df, 
            POPULATION_FEATURE, 
            filter_single_gene=False
        )
        del raw_df
        transformations = pd.read_csv(input[1])
        transformed_df = feature_transform(
            clean_df,
            transformations,
            CHANNELS
        )
        del clean_df
        transformed_df.to_hdf(output[0], key='data', mode='w')

# Standardize features
rule standardize_features:
    input:
        'aggregate_4/hdf/transformed_data.hdf'
    output:
        'aggregate_4/hdf/standardized_data.hdf'
    resources:
        mem_mb=500000
    priority: 1
    run:
        df = pd.read_hdf(input[0])
        feature_start_idx = df.columns.get_loc(FEATURE_START)
        target_features = df.columns[feature_start_idx:].tolist()
        
        standardized_df = grouped_standardization(
            df,
            population_feature=POPULATION_FEATURE,
            control_prefix=CONTROL_PREFIX,
            group_columns=GROUP_COLUMNS,
            index_columns=INDEX_COLUMNS,
            cat_columns=CAT_COLUMNS,
            target_features=target_features,
            drop_features=False
        )
        
        standardized_df.to_hdf(output[0], key='data', mode='w')

# rule concatenate data -- missing for now

# Split mitotic and interphase data
rule split_phases:
    input:
        'aggregate_4/hdf/standardized_data.hdf'
    output:
        'aggregate_4/hdf/mitotic_data.hdf',
        'aggregate_4/hdf/interphase_data.hdf'
    resources:
        mem_mb=500000
    priority: 1
    run:
        df = pd.read_hdf(input[0])
        mitotic_df, interphase_df = split_mitotic_simple(df, THRESHOLDS)
        
        mitotic_df.to_hdf(output[0], key='data', mode='w')
        interphase_df.to_hdf(output[1], key='data', mode='w')

# Create filename-enhanced dataframes
rule prepare_mitotic_montage_data:
    input:
        'aggregate_4/hdf/mitotic_data.hdf',
    output:
        'aggregate_4/hdf/mitotic_montage_data.hdf',
    resources:
        mem_mb=800000
    run:
        # Process mitotic data
        df_mitotic = pd.read_hdf(input[0])
        df_mitotic = add_filenames(df_mitotic, base_ph_file_path='input_ph_tif', 
                                 multichannel_dict=CHANNEL_DICT, subset=True)
        df_mitotic.to_hdf(output[0], key='data', mode='w')

# Create filename-enhanced dataframes
rule prepare_interphase_montage_data:
    input:
        'aggregate_4/hdf/interphase_data.hdf',
    output:
        'aggregate_4/hdf/interphase_montage_data.hdf',
    resources:
        mem_mb=800000
    run:
        # Process interphase data
        df_interphase = pd.read_hdf(input[0])
        df_interphase = add_filenames(df_interphase, base_ph_file_path='input_ph_tif', 
                                 multichannel_dict=CHANNEL_DICT, subset=True)
        df_interphase.to_hdf(output[0], key='data', mode='w')

# Generate montages
rule generate_mitotic_montage:
    input:
        'aggregate_4/hdf/mitotic_montage_data.hdf'
    output:
        'montage/mitotic/{gene}_{sgrna}_{channel}.tif'
    run:
        df = pd.read_hdf(input[0])
        
        montage = create_sgrna_montage(
            df,
            wildcards.gene,
            wildcards.sgrna,
            wildcards.channel,
            PLOTTING_DICT)
        
        if montage is not None:
            save(output[0], montage,
                 display_mode='grayscale',
                 display_ranges=DISPLAY_RANGES[wildcards.channel])
        else:
            # Create empty file if no cells found
            open(output[0], 'w').close()

# Generate montages
rule generate_interphase_montage:
    input:
        'aggregate_4/hdf/interphase_montage_data.hdf'
    output:
        'montage/interphase/{gene}_{sgrna}_{channel}.tif'
    resources:
        mem_mb=20000
    run:
        df = pd.read_hdf(input[0])

        montage = create_sgrna_montage(
            df,
            wildcards.gene,
            wildcards.sgrna,
            wildcards.channel,
            PLOTTING_DICT)
        
        if montage is not None:
            save(output[0], montage,
                 display_mode='grayscale',
                 display_ranges=DISPLAY_RANGES[wildcards.channel])
        else:
            # Create empty file if no cells found
            open(output[0], 'w').close()
    
# Process mitotic gene data
rule process_mitotic_gene_data:
    input:
        'aggregate_4/hdf/mitotic_data.hdf'
    output:
        'aggregate_4/csv/mitotic_gene_data.csv'
    resources:
        mem_mb=100000
    priority: 1
    run:
        df = pd.read_hdf(input[0])
        feature_start_idx = df.columns.get_loc(FEATURE_START)
        target_features = df.columns[feature_start_idx:].tolist()
        
        standardized_df = grouped_standardization(
            df,
            population_feature=POPULATION_FEATURE,
            control_prefix=CONTROL_PREFIX,
            group_columns=GROUP_COLUMNS,
            index_columns=INDEX_COLUMNS,
            cat_columns=CAT_COLUMNS,
            target_features=target_features,
            drop_features=True
        )
        del df
        
        sgrna_df = collapse_to_sgrna(
            standardized_df,
            method='median',
            target_features=target_features,
            index_features=[POPULATION_FEATURE, 'sgRNA_0'],
            control_prefix=CONTROL_PREFIX
        )
        del standardized_df

        gene_df = collapse_to_gene(
            sgrna_df,
            target_features=target_features,
            index_features=[POPULATION_FEATURE]
        )
        del sgrna_df
        
        gene_df.to_csv(output[0], index=True)

# Process interphase gene data
rule process_interphase_gene_data:
    input:
        'aggregate_4/hdf/interphase_data.hdf'
    output:
        'aggregate_4/csv/interphase_gene_data.csv'
    resources:
        mem_mb=500000
    priority: 1
    run:
        df = pd.read_hdf(input[0])
        feature_start_idx = df.columns.get_loc(FEATURE_START)
        target_features = df.columns[feature_start_idx:].tolist()
        
        standardized_df = grouped_standardization(
            df,
            population_feature=POPULATION_FEATURE,
            control_prefix=CONTROL_PREFIX,
            group_columns=GROUP_COLUMNS,
            index_columns=INDEX_COLUMNS,
            cat_columns=CAT_COLUMNS,
            target_features=target_features,
            drop_features=True
        )
        del df
        
        sgrna_df = collapse_to_sgrna(
            standardized_df,
            method='median',
            target_features=target_features,
            index_features=[POPULATION_FEATURE, 'sgRNA_0'],
            control_prefix=CONTROL_PREFIX
        )
        del standardized_df

        gene_df = collapse_to_gene(
            sgrna_df,
            target_features=target_features,
            index_features=[POPULATION_FEATURE]
        )
        del sgrna_df
        
        gene_df.to_csv(output[0], index=True)

# Process all gene data
rule process_all_gene_data:
    input:
        'aggregate_4/hdf/standardized_data.hdf'
    output:
        'aggregate_4/csv/all_gene_data.csv'
    resources:
        mem_mb=300000
    priority: 1
    run:
        df = pd.read_hdf(input[0])
        feature_start_idx = df.columns.get_loc(FEATURE_START)
        target_features = df.columns[feature_start_idx:].tolist()
        
        sgrna_df = collapse_to_sgrna(
            df,
            method='median',
            target_features=target_features,
            index_features=[POPULATION_FEATURE, 'sgRNA_0'],
            control_prefix=CONTROL_PREFIX
        )
        del df

        gene_df = collapse_to_gene(
            sgrna_df,
            target_features=target_features,
            index_features=[POPULATION_FEATURE]
        )
        del sgrna_df
        
        gene_df.to_csv(output[0], index=True)