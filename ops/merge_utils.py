"""
Utility functions for merge analysis pipeline.
Contains functions for FOV calculations, gene mapping, and data cleaning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

def fov_distance(df: pd.DataFrame, 
                i: str = 'i',
                j: str = 'j',
                dimensions: tuple = (2960, 2960),
                suffix: str = '') -> pd.DataFrame:
    """
    Calculate the distance of each cell from the center of the field of view.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing position coordinates
    i : str
        Column name for x-coordinate
    j : str
        Column name for y-coordinate
    dimensions : tuple
        Tuple of (width, height) for the field of view
    suffix : str
        Suffix to append to the output column name
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'fov_distance{suffix}' column
    """
    distance = lambda x: np.sqrt((x[i] - (dimensions[0]/2))**2 + 
                                (x[j] - (dimensions[1]/2))**2)
    df[f'fov_distance{suffix}'] = df.apply(distance, axis=1)
    return df

def identify_single_gene_mappings(x: pd.Series,
                                gene_symbol_0: str = 'gene_symbol_0',
                                gene_symbol_1: str = 'gene_symbol_1') -> bool:
    """
    Determine if a row has a single mapped gene based on gene symbols.
    
    Parameters
    ----------
    x : pd.Series
        Single row from DataFrame containing gene symbol columns
    gene_symbol_0 : str
        Column name of the first gene symbol
    gene_symbol_1 : str
        Column name of the second gene symbol
    
    Returns
    -------
    bool
        True if only gene_symbol_0 exists or both symbols are identical
    """
    has_single_gene = (pd.notnull(x[gene_symbol_0]) & pd.isnull(x[gene_symbol_1])) 
    has_matching_genes = (x[gene_symbol_0] == x[gene_symbol_1])
    return has_single_gene or has_matching_genes

def calculate_channel_mins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minimum values across all channel columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing channel columns with '_min' suffix
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'channels_min' column
    """
    min_cols = [col for col in df.columns if '_min' in col]
    df['channels_min'] = df[min_cols].apply(lambda x: x.min(axis=0), axis=1)
    return df

def plot_cell_positions(df, title, color=None, hue='channels_min'):
    """
    Generate cell position plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing cell position data with i_0, j_0 columns
    title : str
        Plot title
    color : str, optional
        Fixed color for all points
    hue : str, optional
        Column name for color variation (default: 'channels_min')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    fig = plt.figure(figsize=(20, 20))
    
    # If color is specified, override hue
    if color is not None:
        sns.scatterplot(data=df, x='i_0', y='j_0', color=color, alpha=0.5)
    else:
        sns.scatterplot(data=df, x='i_0', y='j_0', hue=hue, alpha=0.5)
        
    plt.title(title)
    plt.xlabel('i_0')
    plt.ylabel('j_0')
    return fig

def plot_channel_histogram(df_before, df_after):
    """
    Generate histogram of channel values with density normalization 
    and consistent bin edges.
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Calculate bin edges based on the full range of data
    min_val = min(df_before['channels_min'].min(), df_after['channels_min'].min())
    max_val = max(df_before['channels_min'].max(), df_after['channels_min'].max())
    bins = np.linspace(min_val, max_val, 51)  # 51 edges makes 50 bins
    
    # Plot normalized histograms with consistent bins
    plt.hist(df_before['channels_min'].dropna(), bins=bins,
             density=True, color='blue', alpha=0.5, label='Before clean')
    plt.hist(df_after['channels_min'].dropna(), bins=bins,
             density=True, color='orange', alpha=0.5, label='After clean')
    
    plt.title('Histogram of channels_min Values')
    plt.xlabel('channels_min')
    plt.ylabel('Density')
    plt.legend()
    return fig

def deduplicate_cells(df, mapped_single_gene=False, return_stats=False):
    """
    Remove duplicate cell mappings in two steps:
    1. For each phenotype cell (cell_0), keep the best SBS cell match
    2. For each SBS cell (cell_1), keep the best phenotype cell match
    
    Args:
        df: DataFrame with merged cell data
    
    Returns:
        DataFrame with duplicates removed
    """
    # Step 1: For each phenotype cell, keep best SBS match
    # Sort by mapping quality and distance to prioritize better matches
    df_sbs_deduped = (df
        .sort_values(['mapped_single_gene', 'fov_distance_1'], 
                    ascending=[False, True])
        .drop_duplicates(['well', 'tile', 'cell_0'], 
                        keep='first'))
    
    # Step 2: For each remaining SBS cell, keep best phenotype match
    df_final = (df_sbs_deduped
        .sort_values('fov_distance_0', ascending=True)
        .drop_duplicates(['well', 'site', 'cell_1'], 
                        keep='first'))
    
    # Calculate statistics
    stats = {
        'stage': ['Initial', 'After SBS dedup', 'After phenotype dedup'],
        'total_cells': [len(df), len(df_sbs_deduped), len(df_final)],
        'mapped_genes': [
            df[df.mapped_single_gene==True].pipe(len),
            df_sbs_deduped[df_sbs_deduped.mapped_single_gene==True].pipe(len),
            df_final[df_final.mapped_single_gene==True].pipe(len)
        ]
    }
    
    # Print summary statistics
    print(f"Initial cells: {stats['total_cells'][0]:,}")
    print(f"After SBS deduplication: {stats['total_cells'][1]:,}")
    print(f"After phenotype deduplication: {stats['total_cells'][2]:,}")
    print(f"Final mapped genes: {stats['mapped_genes'][2]:,}")

    if mapped_single_gene:
        print("\nFilter to cells with single gene mappings.")
        df_final = df_final[df_final.mapped_single_gene]
    else:
        print("\nKeeping all deduped cells.")

    if return_stats:
        return df_final, pd.DataFrame(stats)
    else:
        return df_final

def check_matching_rates(orig_data, merged_data, modality='sbs', return_stats=False):
    """
    Check what fraction of original cells survived the merging/cleaning process.
    
    Args:
        orig_data: Original dataset (sbs_info or phenotype_info)
        merged_data: Cleaned/merged dataset to compare against
        modality: Either 'sbs' or 'phenotype'
    """
    # Set up merge parameters based on modality
    if modality == 'sbs':
        merge_cols = ['well', 'site', 'cell_1']
        rename_dict = {'cell': 'cell_1', 'tile': 'site'}
    else:
        merge_cols = ['well', 'tile', 'cell_0']
        rename_dict = {'cell': 'cell_0'}
    
    # Prepare and merge data
    checking_df = (orig_data
        .rename(columns=rename_dict)
        .drop(columns=['i', 'j'])
        .merge(merged_data, how='left', on=merge_cols))
    
    # Calculate matching rates per well
    rates = []
    print(f"\nFinal matching rates for {modality.upper()} cells:")
    for well, df in checking_df.groupby('well'):
        total = len(df)
        matched = df.distance.notna().sum()
        rate = matched / total * 100
        print(f"Well {well}: {rate:.1f}% ({matched:,}/{total:,} cells)")
        
        rates.append({
            'well': well,
            'total_cells': total,
            'matched_cells': matched,
            'match_rate': rate
        })

    rates_df = pd.DataFrame(rates)

    if return_stats:
        return rates_df
