#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ops.qc import *
from ops.merge_utils import *

# Set screen directories
ph_function_home = "/lab/barcheese01/screens"
ph_function_dataset = "denali"
home = os.path.join(ph_function_home, ph_function_dataset)

# Create directories if they don't exist
qc_dir = os.path.join(home, 'merge_3', 'qc')
os.makedirs(qc_dir, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(qc_dir, filename))
    plt.close(fig)

# Optional misalignment parameters
misaligned_wells = None
misaligned_tiles = None

# Read merged data
print("Reading initial merged dataset...")
df_merged = pd.read_hdf(os.path.join(home, 'merge_3', 'hdf', 'merge_all.hdf'))
print(f"Initial merged data shape: {df_merged.shape}")

# Calculate FOV distances for both imaging modalities
print("\nCalculating FOV distances...")
print("- For ph imaging FOVs (2960x2960)")
df_merged = df_merged.pipe(fov_distance, i='i_0', j='j_0', dimensions=(2960, 2960), suffix='_0')
print("- For sbs imaging FOVs (1480x1480)")
df_merged = df_merged.pipe(fov_distance, i='i_1', j='j_1', dimensions=(1480, 1480), suffix='_1')

print("\nClosest cells to FOV center (ph imaging):")
print(df_merged.sort_values(['fov_distance_0']).head(10))

print("\nClosest cells to FOV center (sbs imaging):")
print(df_merged.sort_values(['fov_distance_1']).head(10))

# Process gene mapping from sbs data
print("\nProcessing gene mapping from sbs data...")
df_sbs = pd.read_hdf(os.path.join(home, 'sbs_1', 'hdf', 'cells.hdf'))
df_sbs['mapped_single_gene'] = df_sbs.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)

# Process gene mapping statistics
print("\Gene mapping statistics:")
mapping_counts = df_sbs.mapped_single_gene.value_counts()
mapping_percentages = df_sbs.mapped_single_gene.value_counts(normalize=True)

print("\nCounts:")
print(f"Uniquely mapped cells: {mapping_counts[True]:,}")
print(f"Unmapped/not uniquely mapped cells: {mapping_counts[False]:,}")
print(f"Total cells (from sbs data): {len(df_sbs):,}")

print("\nPercentages:")
print(f"Uniquely mapped cells: {mapping_percentages[True]:.2%}")
print(f"Unmapped/not uniquely mapped cells: {mapping_percentages[False]:.2%}")

# Save cell mapping statistics
mapping_stats = pd.DataFrame({
    'category': ['mapped_cells', 'unmapped_cells'],
    'count': [mapping_counts[True], mapping_counts[False]],
    'percentage': [mapping_percentages[True], mapping_percentages[False]]
})
mapping_stats.to_csv(os.path.join(qc_dir, 'cell_mapping_stats.csv'), index=False)

# Merge cell information from sbs
print("\nMerging gene mapping information from sbs...")
df_merged = df_merged.merge(
    df_sbs[['well', 'tile', 'cell', 'sgRNA_0', 'gene_symbol_0', 'mapped_single_gene']]
    .rename({'tile': 'site', 'cell': 'cell_1'}, axis=1),
    how='left',
    on=['well', 'site', 'cell_1']
)

print("\nSample of merged data with gene mapping (sorted by mapping and distance):")
print(df_merged.head(10).sort_values(
    ['mapped_single_gene', 'fov_distance_1'],
    ascending=[False, True]
))

# Process cell intensity information from ph data
print("\nProcessing cell intensity information from ph data...")
df_phenotype = pd.read_hdf(os.path.join(home, 'ph_2', 'hdf', 'min_cp_phenotype.hdf'))
print(f"Total cells (from ph data): {len(df_phenotype):,}")

# Calculate minimum channel values for cells
df_phenotype = calculate_channel_mins(df_phenotype)

# Print summary of cell channel minimums
print("\nCell intensity minimum value statistics:")
print(df_phenotype['channels_min'].describe())

# Merge cell information from ph
print("\nMerging cell intensity information from ph data...")
df_merged = df_merged.merge(
    df_phenotype[['well', 'tile', 'label', 'channels_min']]
    .rename(columns={'label': 'cell_0'}),
    how='left',
    on=['well', 'tile', 'cell_0']
)

print("\nMerged data shape:", df_merged.shape)
print("\nSample of merged data with cell intensities:")
print(df_merged[['well', 'tile', 'cell_0', 'mapped_single_gene', 'channels_min']]
    .sort_values(['channels_min'])
    .head(10))

# Run matching analysis before cleanup
print("\nAnalyzing matching rates between ph and sbs data...")
df_merged_minimal = df_merged[['well', 'tile', 'site', 'cell_0', 'cell_1', 'distance']]

# For SBS matching rates
print("\nCalculating SBS to PH matching rates...")
plt.figure(figsize=(12, 6))
sbs_summary, _ = plot_sbs_ph_matching_heatmap(
    df_merged_minimal,  # df_merged_minimal: contains matched cells
    df_sbs,     # df_info: all SBS cells
    target='sbs',
    shape='6W_sbs',
    return_summary=True
)
save_plot(plt.gcf(), 'sbs_to_ph_matching_rates.png')

print("\nSBS matching rate summary:")
print(sbs_summary.groupby('well')['fraction of sbs cells matched to phenotype cells'].describe())

# For PH matching rates
print("\nCalculating PH to SBS matching rates...")
plt.figure(figsize=(12, 6))
ph_summary, _ = plot_sbs_ph_matching_heatmap(
    df_merged_minimal,    # df_merged_minimal: contains matched cells
    df_phenotype.rename(columns={'label': 'cell_0'}), # df_info: all PH cells
    target='phenotype',
    shape='6W_ph',
    return_summary=True
)
save_plot(plt.gcf(), 'ph_to_sbs_matching_rates.png')

print("\nPH matching rate summary:")
print(ph_summary.groupby('well')['fraction of phenotype cells matched to sbs cells'].describe())

# Save matching rate data
sbs_summary.to_csv(os.path.join(qc_dir, 'sbs_matching_rates.csv'), index=False)
ph_summary.to_csv(os.path.join(qc_dir, 'ph_matching_rates.csv'), index=False)

# Create a combined summary
print("\nOverall matching statistics:")
print(f"Average SBS matching rate: {sbs_summary['fraction of sbs cells matched to phenotype cells'].mean():.2%}")
print(f"Average PH matching rate: {ph_summary['fraction of phenotype cells matched to sbs cells'].mean():.2%}")

# Continue with cleanup and saving
del df_merged_minimal, df_sbs, df_phenotype

# Save merged data
df_merged.to_hdf(os.path.join(home, 'merge_3', 'hdf', 'merge_all_stats.hdf'), 'x', mode='w')

if misaligned_wells is not None and misaligned_tiles is not None:
    print("\nExcluding misaligned wells/tiles:")
    print(f"Misaligned wells: {misaligned_wells}")
    print(f"Misaligned tiles: {misaligned_tiles}")
else:
    print("\nNo misaligned wells/tiles specified")

print("\nGenerating visualization plots of cells per tile...")
print("1. Plotting all cells colored by channel_min value...")
plt.figure(figsize=(20, 20))
plot_cell_positions(df_merged, title='All Cells by Channel Min')
save_plot(plt.gcf(), 'all_cells_by_channel_min.png')

print("2. Plotting cells with channel_min = 0 (potential edge effects)...")
plt.figure(figsize=(20, 20))
plot_cell_positions(df_merged.query('channels_min==0'), 
                   title='Cells with Channel Min = 0',
                   color='red')
save_plot(plt.gcf(), 'cells_with_channel_min_0.png')

if misaligned_wells is not None and misaligned_tiles is not None:
    print("3. Plotting cells excluding misaligned wells/tiles...")
    plt.figure(figsize=(20, 20))
    plot_cell_positions(df_merged.query('well != @misaligned_wells & tile != @misaligned_tiles'),
                        title='Cells with Channel Min > 0 (excluding misaligned wells/tiles)')
    save_plot(plt.gcf(), 'cells_with_channel_min_gt_0_excl_misaligned.png')

print("\nCleaning merged dataset...")
print(f"Initial number of cells: {len(df_merged):,}")

# Create cleaned dataset by removing cells with channel_min == 0
df_merged_clean = df_merged.query('channels_min>0')

# If misaligned wells/tiles specified, exclude them
if misaligned_wells is not None and misaligned_tiles is not None:
    df_merged_clean = df_merged_clean.query('well != @misaligned_wells & tile != @misaligned_tiles')

print(f"Final number of cells: {len(df_merged_clean):,} ({len(df_merged_clean)/len(df_merged):.1%} retained)")

# Generate and save histogram
print("\nGenerating channel value distribution histogram...")
plt.figure(figsize=(12, 6))
plot_channel_histogram(df_merged, df_merged_clean)
save_plot(plt.gcf(), 'channel_min_histogram.png')

# Save cleaned dataset
print("\nSaving cleaned dataset...")
df_merged_clean.to_hdf(os.path.join(home, 'merge_3', 'hdf', 'merge_all_clean.hdf'), 'x', mode='w')

# Clean up memory
del df_merged

print("Starting deduplication process...")
df_merged_deduped, deduped_stats = deduplicate_cells(df_merged_clean, mapped_single_gene=False, return_stats=True)
deduped_stats.to_csv(os.path.join(qc_dir, 'deduplication_stats.csv'), index=False)
df_merged_deduped.to_hdf(os.path.join(home, 'merge_3', 'hdf', 'merged_deduped.hdf'), 'x', mode='w')

print("\nChecking cell matching rates after deduplication...")
# Load info files for matching rate checks
df_sbs_info = pd.read_hdf(os.path.join(home, 'sbs_1', 'hdf', 'sbs_info.hdf'))
print(f"Total cells (from sbs info): {len(df_sbs_info):,}")
df_phenotype_info = pd.read_hdf(os.path.join(home, 'ph_2', 'hdf', 'phenotype_info.hdf'))
print(f"Total cells (from ph info): {len(df_phenotype_info):,}")

# Get matching rates statistics
sbs_rates = check_matching_rates(df_sbs_info, df_merged_deduped, modality='sbs', return_stats=True)
sbs_rates.to_csv(os.path.join(qc_dir, 'final_sbs_matching_rates.csv'), index=False)
ph_rates = check_matching_rates(df_phenotype_info, df_merged_deduped, modality='phenotype', return_stats=True)
ph_rates.to_csv(os.path.join(qc_dir, 'final_phenotype_matching_rates.csv'), index=False)

# Clean up
del df_sbs_info, df_phenotype_info

print("\nMerging with full phenotype data...")
# Load and merge with full phenotype data
df_cp_phenotype = pd.read_hdf(os.path.join(home, 'ph_2', 'hdf', 'cp_phenotype.hdf'))
df_final = df_merged_deduped.merge(
    df_cp_phenotype.rename(columns={'label':'cell_0'}),
    how='left',
    on=['well', 'tile', 'cell_0']
)

# Save final merged dataset
print("Saving final merged dataset...")
df_final.to_hdf(os.path.join(home, 'merge_3', 'hdf', 'merged_final.hdf'), 'x', mode='w')
