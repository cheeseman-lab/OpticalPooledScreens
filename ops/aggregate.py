"""
Single-Cell Data Aggregation Utilities
This module provides functions for processing, transforming, and analyzing single-cell data
across different granularities (relating to aggregate -- step 4). It includes functions for:

1. Data Loading: Tools for efficient loading and sampling of large HDF datasets.
2. Feature Processing: Functions for transforming and standardizing feature measurements.
3. Population Analysis: Methods for identifying and separating cell populations (e.g., mitotic vs interphase).
4. Data Aggregation: Tools for collapsing data from cell-level to sgRNA and gene-level summaries.
5. Quality Control: Functions for parameter suggestion and visualization of distributions.

"""

import numpy as np
import pandas as pd
import os
import ops.utils as utils
import ops.io as io
import ops.annotate as annotate
from ops.io import save_stack as save
import matplotlib.pyplot as plt
import h5py
import seaborn as sns

def load_hdf_subset(file_path, n_rows=20000, population_feature='gene_symbol_0'):
    """
    Load a fixed number of random rows from an HDF file without loading entire file into memory.
    
    Parameters
    ----------
    file_path : str
        Path to HDF file
    n_rows : int
        Number of rows to get
    
    Returns
    -------
    pd.DataFrame
        Subset of the data with combined blocks
    """
    print(f"Reading first {n_rows:,} rows from {file_path}")

    # read the first n_rows of the file path
    df = pd.read_hdf(file_path, stop=n_rows)

    # print the number of unique populations
    print(f"Unique populations: {df[population_feature].nunique()}")

    # print the counts of the well variable
    print(df["well"].value_counts())

    return df

def explode_with_plotting_dict(df_barcodes, plotting_dict):
    """
    Explode DataFrame to include all channels from plotting_dict.
    Handles both simple and nested channel dictionaries.
    
    Args:
        df_barcodes (pd.DataFrame): DataFrame with gene/sgRNA combinations
        plotting_dict (dict): Dictionary mapping channels to filename columns
            Can be simple (str) or nested (dict with 'filename' and optional 'channel')
    """
    channels = list(plotting_dict.keys())
    return (df_barcodes
            .assign(channel=lambda x: [channels]*len(x))
            .explode('channel'))

def clean_cell_data(df, population_feature, filter_single_gene=False):
   """
   Clean cell data by removing cells without perturbation assignments and optionally filtering for single-gene cells.
   
   Args:
       df (pd.DataFrame): Raw dataframe containing cell measurements
       population_feature (str): Column name containing perturbation assignments
       filter_single_gene (bool): If True, only keep cells with mapped_single_gene=True
   
   Returns:
       pd.DataFrame: Cleaned dataframe
   """
   # Remove cells without perturbation assignments
   clean_df = df[df[population_feature].notna()].copy()
   print(f"Found {len(clean_df)} cells with assigned perturbations")
   
   if filter_single_gene:
       # Filter for single-gene cells if requested
       clean_df = clean_df[clean_df['mapped_single_gene'] == True]
       print(f"Kept {len(clean_df)} cells with single gene assignments")
   else:
       # Warn about multi-gene cells if not filtering
       multi_gene_cells = len(clean_df[clean_df['mapped_single_gene'] == False])
       if multi_gene_cells > 0:
           print(f"WARNING: {multi_gene_cells} cells have multiple gene assignments")
   
   return clean_df

def feature_transform(df, transformation_dict, channels):
    """
    Apply transformations to features based on a transformation dictionary.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features to transform
    transformation_dict : pd.DataFrame
        DataFrame containing 'feature' and 'transformation' columns specifying
        which transformations to apply to which features
    channels : list
        List of channel names to use when expanding feature templates
        
    Returns
    -------
    pd.DataFrame
        DataFrame with transformed features
    """
    def apply_transformation(feature, transformation):
        if transformation == 'log(feature)':
            return np.log(feature)
        elif transformation == 'log(feature-1)':
            return np.log(feature - 1)
        elif transformation == 'log(1-feature)':
            return np.log(1 - feature)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    df = df.copy()
    
    for _, row in transformation_dict.iterrows():
        feature_template = row['feature']
        transformation = row['transformation']
        
        # Handle single channel features
        if '{channel}' in feature_template:
            for channel in channels:
                feature = feature_template.replace("{channel}", channel)
                if feature in df.columns:
                    df[feature] = apply_transformation(df[feature], transformation)
        
        # Handle double channel features (overlap)
        elif '{channel1}' in feature_template and '{channel2}' in feature_template:
            for channel1 in channels:
                for channel2 in channels:
                    if channel1 != channel2:
                        feature = feature_template.replace("{channel1}", channel1).replace("{channel2}", channel2)
                        if feature in df.columns:
                            df[feature] = apply_transformation(df[feature], transformation)
    
    return df

def grouped_standardization(df, population_feature='gene_symbol_0', control_prefix='sg_nt', 
                          group_columns=['well'], index_columns=['tile', 'cell_0'],
                          cat_columns=['gene_symbol_0', 'sgRNA_0'], target_features=None, 
                          drop_features=False):
    """
    Standardize features using robust z-scores, calculated per group using control populations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    population_feature : str
        Column name containing population identifiers
    control_prefix : str
        Prefix identifying control populations
    group_columns : list
        Columns to group by for standardization
    index_columns : list
        Columns that uniquely identify cells
    cat_columns : list
        Categorical columns to preserve
    target_features : list, optional
        Features to standardize. If None, will standardize all numeric columns
    drop_features : bool
        Whether to drop untransformed features
        
    Returns
    -------
    pd.DataFrame
        Standardized dataframe
    """
    df_out = df.copy().drop_duplicates(subset=group_columns + index_columns)

    if target_features is None:
        target_features = [col for col in df.columns 
                         if col not in group_columns + index_columns + cat_columns]
    
    if drop_features:
        df = df[group_columns + index_columns + cat_columns + target_features]

    unstandardized_features = [col for col in df.columns if col not in target_features]
    
    # Filter control group
    control_group = df[df[population_feature].str.startswith(control_prefix)]
    
    # Calculate MAD
    def median_absolute_deviation(arr):
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return mad
    
    # Calculate group statistics
    group_medians = control_group.groupby(group_columns)[target_features].median()
    group_mads = control_group.groupby(group_columns)[target_features].apply(
        lambda x: x.apply(median_absolute_deviation))

    # Standardize using robust z-score
    df_out = pd.concat([
        df_out[unstandardized_features].set_index(group_columns + index_columns),
        df_out.set_index(group_columns + index_columns)[target_features]
            .subtract(group_medians)
            .divide(group_mads)
            .multiply(0.6745)  # Scale factor for robust z-score
    ], axis=1)

    return df_out.reset_index()

def add_filenames(df, base_ph_file_path=None, multichannel_dict=None, subset=False):
    """
    Add filename columns to DataFrame for single or multiple channels.
    
    Args:
        df (pd.DataFrame): DataFrame containing well and tile columns
        base_ph_file_path (str, optional): Base path to the ph image directory (of converted tiff files)
        multichannel_dict (dict, optional): Dictionary mapping channel names to filename suffixes
            Example: {
                'A594': 'Channel-A594_1x1_LF.tif',
                'A750': 'Channel-A750_1x1_LF.tif',
                'DAPI': 'Channel-DAPI_1x1_LF.tif',
                'GFP': 'Channel-GFP_1x1_LF.tif'
            }
    
    Returns:
        pd.DataFrame: DataFrame with new filename columns for each channel
    """      
    df = df.copy()
    
    def generate_filename(row, channel_suffix=None):
        # Basic filename pattern
        base = f"20X_{row['well']}_Tile-{row['tile']}"
        
        if channel_suffix:
            # Multichannel format
            return f"{base_ph_file_path}/{base}.phenotype.{channel_suffix}"
        else:
            # Single channel format
            return f"{base_ph_file_path}/{base}.phenotype.tif"
    
    if multichannel_dict is not None:
        # Add a filename column for each channel
        for channel_name, suffix in multichannel_dict.items():
            col_name = f'filename_{channel_name}'
            df[col_name] = df.apply(lambda row: generate_filename(row, suffix), axis=1)
    else:
        # Single filename column for non-multichannel case
        df['filename'] = df.apply(lambda row: generate_filename(row), axis=1)

    # Subset to only data that is required for montage generation
    if subset:
        essential_columns = [
            'gene_symbol_0',  
            'sgRNA_0',       
            'well',          
            'tile',          
            'i_0',          
            'j_0',
        ]
        # Add filename columns
        if multichannel_dict is not None:
            essential_columns.extend([f'filename_{channel}' for channel in multichannel_dict.keys()])
        else:
            essential_columns.append('filename')
            
        # Only keep columns we need
        df = df[essential_columns]
    
    return df

def split_mitotic_simple(df, conditions):
    """
    Split cells into mitotic and interphase populations based on feature thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    conditions : dict
        Dictionary mapping feature names to (threshold, direction) tuples
        where direction is 'greater' or 'less'
        
    Returns
    -------
    tuple
        (mitotic_df, interphase_df) pair of DataFrames
    """
    mitotic_df = df.copy()
    
    for feature, (cutoff, direction) in conditions.items():
        if direction == 'greater':
            mitotic_df = mitotic_df[mitotic_df[feature] > cutoff]
        elif direction == 'less':
            mitotic_df = mitotic_df[mitotic_df[feature] < cutoff]
        else:
            raise ValueError("Direction must be 'greater' or 'less'")
    
    interphase_df = df.drop(mitotic_df.index)
    
    return mitotic_df, interphase_df

def collapse_to_sgrna(df, method='median', target_features=None, 
                     index_features=['gene_symbol_0', 'sgRNA_0'],
                     control_prefix='sg_nt', min_count=None):
    """
    Collapse cell-level data to sgRNA-level summaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with cell-level data
    method : str
        Method for collapsing ('median' only currently supported)
    target_features : list, optional
        Features to collapse. If None, uses all numeric columns
    index_features : list
        Columns that identify sgRNAs
    control_prefix : str
        Prefix identifying control sgRNAs
    min_count : int, optional
        Minimum number of cells required per sgRNA
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sgRNA-level summaries
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    if method == 'median':
        df_out = df.groupby(index_features)[target_features].median().reset_index()
        df_out['sgrna_count'] = df.groupby(index_features).size().reset_index(
            name='sgrna_count')['sgrna_count']
        
        if min_count is not None:
            df_out = df_out.query('sgrna_count >= @min_count')
    else:
        raise ValueError("Only method='median' is currently supported")

    control_mask = df_out['gene_symbol_0'].str.startswith(control_prefix)
    unique_controls = df_out.loc[control_mask, 'gene_symbol_0'].unique()
    if len(unique_controls) == 1: 
        print("Multiple control guides not found. Renaming to ensure uniqueness.")
        control_mask = df_out['gene_symbol_0'].str.startswith(control_prefix)
        for idx, row_idx in enumerate(df_out[control_mask].index, 1):
            df_out.loc[row_idx, 'gene_symbol_0'] = f"{control_prefix}_{idx}"
    
    return df_out


def collapse_to_gene(df, target_features=None, index_features=['gene_symbol_0'], min_count=None):
    """
    Collapse sgRNA-level data to gene-level summaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with sgRNA-level data
    target_features : list, optional
        Features to collapse. If None, uses all numeric columns
    index_features : list
        Columns that identify genes
    min_count : int, optional
        Minimum number of sgRNAs required per gene
        
    Returns
    -------
    pd.DataFrame
        DataFrame with gene-level summaries
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    df_out = df.groupby(index_features)[target_features].median().reset_index()

    if 'sgrna_count' in df.columns:
        df_out['gene_count'] = df.groupby(index_features)['sgrna_count'].sum().reset_index(
            drop=True)

    if min_count is not None:
        df_out = df_out.query('gene_count >= @min_count')
        
    return df_out

def suggest_parameters(df, population_feature):
    """
    Suggest parameters based on input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    population_feature : str
        Column name containing population identifiers

    Returns
    -------
    None
    """ 
    
    # Look for potential control prefixes
    unique_populations = df[population_feature].unique()
    potential_controls = [pop for pop in unique_populations if any(
        control in pop.lower() 
        for control in ['nt', 'non-targeting', 'control', 'ctrl', 'neg']
    )]
    
    # Find first feature-like column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    potential_features = [col for col in numeric_cols if any(
        pattern in col.lower() 
        for pattern in ['mean', 'median', 'std', 'intensity', 'area']
    )]
    
    # Identify metadata-like columns
    potential_metadata = df.select_dtypes(include=['object']).columns.tolist()
    
    print("\nSuggested Parameters:")
    print("-" * 50)
    
    if potential_controls:
        print("\nPotential control prefixes found:")
        for ctrl in potential_controls:
            print(f"  - '{ctrl}'")
    else:
        print("\nNo obvious control prefixes found. Please check your data.")
        
    if potential_features:
        print(f"\nFirst few feature columns detected:")
        for feat in potential_features[:5]:
            print(f"  - '{feat}'")
            
    print("\nMetadata columns detected:")
    print(f"  - Categorical: {', '.join(potential_metadata[:5])}")

def plot_mitotic_distribution_hist(df, threshold_variable, threshold_value, bins=100):
    """
    Plot distribution of the threshold variable and calculate percent of mitotic cells.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing cell measurements
    threshold_variable : str
        Column name for mitotic cell identification
    threshold_value : float
        Threshold value for separating mitotic cells
    bins : int
        Number of bins for histogram
        
    Returns
    -------
    float
        Percentage of cells classified as mitotic
    """
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(df[threshold_variable], bins=bins)
    plt.title(f'Histogram of {threshold_variable}')
    plt.xlabel(threshold_variable)
    plt.ylabel('Frequency')
    plt.axvline(x=threshold_value, color='r', linestyle='--', 
                label=f'Mitotic threshold ({threshold_value})')
    plt.legend()
    plt.show()
    
    # Calculate percent mitotic
    mitotic_mask = df[threshold_variable] > threshold_value
    percent_mitotic = (mitotic_mask.sum() / len(df)) * 100
    
    print(f"Number of mitotic cells: {mitotic_mask.sum():,}")
    print(f"Total cells: {len(df):,}")
    print(f"Percent mitotic: {percent_mitotic:.2f}%")

def plot_mitotic_distribution_scatter(df, threshold_variable_x, threshold_variable_y, threshold_x, threshold_y,
                                      alpha=0.5):
    """
    Plot scatter plot of two variables with two threshold cutoffs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing cell measurements
    threshold_variable_x : str
        Column name for x-axis variable
    threshold_variable_y : str
        Column name for y-axis variable
    threshold_x : float
        Threshold value for x-axis
    threshold_y : float
        Threshold value for y-axis
    alpha : float
        Transparency of points

    Returns
    -------
    None
    """
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[threshold_variable_x], df[threshold_variable_y], alpha=alpha)
    plt.title(f'Scatter plot of {threshold_variable_x} vs {threshold_variable_y}')
    plt.xlabel(threshold_variable_x)
    plt.ylabel(threshold_variable_y)
    plt.axvline(x=threshold_x, color='r', linestyle='--', 
                label=f'Mitotic threshold ({threshold_x})')
    plt.axhline(y=threshold_y, color='g', linestyle='--',
                label=f'Mitotic threshold ({threshold_y})')
    plt.legend()
    plt.show()

    # Calculate percent mitotic
    mitotic_mask_x = df[threshold_variable_x] > threshold_x
    mitotic_mask_y = df[threshold_variable_y] > threshold_y
    percent_mitotic = (mitotic_mask_x & mitotic_mask_y).sum() / len(df) * 100

    print(f"Number of mitotic cells: {sum(mitotic_mask_x & mitotic_mask_y):,}")
    print(f"Total cells: {len(df):,}")
    print(f"Percent mitotic: {percent_mitotic:.2f}%")
    
def create_mitotic_cell_montage(df,
                                output_dir,
                                output_prefix,
                                channels,
                                display_ranges,
                                num_cells=30,
                                cell_size=40,
                                shape=(3, 10),
                                selection_params=None,
                                coordinate_cols=None
                                ):
    """
    Create a montage of cells from DataFrame with flexible parameters.
    Designed to save data directly, for use with interactive visualization.
    
    Args:
        df (pd.DataFrame): DataFrame with cell data
        output_dir (str): Directory to save montages
        output_prefix (str): Prefix for output filenames
        channels (dict): Dictionary mapping channel names to filename column names
                        e.g. {'DAPI': 'filename_DAPI', 'GFP': 'filename_GFP'}
        display_ranges (dict): Dictionary mapping channel names to display ranges
                        e.g. {'DAPI': [(0, 14000)], 'GFP': [(350, 2000)]}
        num_cells (int): Number of cells to include
        cell_size (int): Size of cell bounds box
        shape (tuple): Shape of montage grid (rows, cols)
        selection_params (dict): Parameters for cell selection
                        {
                            'method': 'random' | 'sorted' | 'head',
                            'sort_by': column name if method='sorted',
                            'ascending': True/False if method='sorted'
                        }
        coordinate_cols (list): Names of coordinate columns for bounds, defaults to ['i_0', 'j_0']
    
    Returns:
        dict: Dictionary mapping channels to their montage arrays
    """ 
    if coordinate_cols is None:
        coordinate_cols = ['i_0', 'j_0']
    
    if selection_params is None:
        selection_params = {'method': 'head'}

    # Select cells based on parameters
    df_subset = df.copy()
    if selection_params['method'] == 'random':
        df_subset = df_subset.sample(n=num_cells)
    elif selection_params['method'] == 'sorted':
        df_subset = (df_subset
                    .sort_values(selection_params['sort_by'], 
                               ascending=selection_params.get('ascending', True))
                    .head(num_cells))
    else:  
        df_subset = df_subset.head(num_cells)
    
    # Add bounds
    df_subset = (df_subset
                .pipe(annotate.add_rect_bounds, 
                     width=cell_size, 
                     ij=coordinate_cols, 
                     bounds_col='bounds'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store montages
    montages = {}
    
    # Create montages for each channel
    for channel_name, channel_info in channels.items():
        # Parse the channel dict 
        if isinstance(channel_info, dict):
            filename = channel_info['filename']
            if filename == 'filename':
                filename = 'filename'
            else:
                filename = f'filename_{filename}'
            channel_idx = channel_info.get('channel')
        else:
            filename = channel_info
            channel_idx = None

        # Create grid
        cell_grid = io.grid_view(
            files=df_subset[filename].tolist(),
            bounds=df_subset['bounds'].tolist(),
            padding=0,
            im_func=None,
            memoize=True
        )

        # Create montage
        montage = utils.montage(cell_grid, shape=shape)
        montages[channel_name] = montage
      
        # Save montage
        output_path = os.path.join(output_dir, f'{output_prefix}_{channel_name}.tif')
        
        # Select channel if specified in channel_info
        if channel_idx is not None:
            montage = montage[channel_idx]
        
        save(output_path, montage,
             display_mode='grayscale',
             display_ranges=display_ranges[channel_name])
        
        print(f"Saved {channel_name} montage to {output_path}")    

def create_sgrna_montage(df,
                        gene,
                        sgrna,
                        channel,
                        plotting_dict,
                        population_feature='gene_symbol_0',
                        sgrna_feature='sgRNA_0',
                        num_cells=50,
                        cell_size=40,
                        shape=(5, 10),
                        coordinate_cols=None):
    """
    Create a montage array for a specific gene-sgRNA-channel combination.
    Designed to work with Snakemake workflows.
    
    Args:
        df (pd.DataFrame): DataFrame with cell data
        gene (str): Gene identifier to filter for
        sgrna (str): sgRNA identifier to filter for
        channel (str): Channel to create montage for
        plotting_dict (dict): Dictionary mapping channels to filename info
            Format: {'channel': {'filename': 'file_column', 'channel': idx}} or
                   {'channel': {'filename': 'file_column'}}
        population_feature (str): Column name for gene identifier
        sgrna_feature (str): Column name for sgRNA identifier
        num_cells (int): Number of cells to include
        cell_size (int): Size of cell bounds box
        shape (tuple): Shape of montage grid (rows, cols)
        coordinate_cols (list): Names of coordinate columns for bounds
    
    Returns:
        np.ndarray: Montage array ready for saving
    """
    if coordinate_cols is None:
        coordinate_cols = ['i_0', 'j_0']
        
    # Filter for specific gene-sgRNA combination
    mask = (df[population_feature] == gene) & (df[sgrna_feature] == sgrna)
    df_subset = df[mask].copy()
    
    if len(df_subset) == 0:
        return None
    
    # Take required number of cells
    df_subset = df_subset.head(num_cells)    
    df_subset = (df_subset
                .pipe(annotate.add_rect_bounds, 
                     width=cell_size, 
                     ij=coordinate_cols, 
                     bounds_col='bounds'))
 
    # Parse channel info
    channel_info = plotting_dict[channel]
    if isinstance(channel_info, dict):
        filename = channel_info['filename']
        if filename == 'filename':
            filename = 'filename'
        else:
            filename = f'filename_{filename}'
        channel_idx = channel_info.get('channel')
    else:
        filename = channel_info
        channel_idx = None
    
    # Create grid
    cell_grid = io.grid_view(
        files=df_subset[filename].tolist(),
        bounds=df_subset['bounds'].tolist(),
        padding=0,
        im_func=None,
        memoize=True
    )

    # Create montage
    montage = utils.montage(cell_grid, shape=shape)
    
    # Select specific channel if needed
    if channel_idx is not None:
        montage = montage[channel_idx]
        
    return montage

def plot_feature_distributions(dfs, features, remove_clean=False):
    """
    Create violin plots with jittered points for feature distributions across different datasets.
    Uses log scale for clean and transformed data, and linear scale for standardized data.
    Points are colored based on control vs non-control status.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary of dataframes with keys as dataset names (e.g., "clean", "transformed")
    features : list
        List of feature names to plot
    remove_clean : bool
        Whether to remove clean data from the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the violin plots
    """
    # Prepare data for plotting
    plot_data = []

    # Remove clean data if requested
    if remove_clean:
        dfs = {key: df for key, df in dfs.items() if key != 'clean'}

    for dataset_name, df in dfs.items():        
        for feature in features:
            feature_data = pd.DataFrame({
                'Dataset': dataset_name,
                'Well': df["well"],
                'Value': df[feature],
                'Feature': feature,
            })
            plot_data.append(feature_data)
    
    # Combine all data
    plot_df = pd.concat(plot_data, ignore_index=True)
    
    # Create subplot for each feature
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(7*n_features, 8))
    
    if n_features == 1:
        axes = [axes]
    
    # Plot violins for each feature
    for ax, feature in zip(axes, features):
        # Create a second axis that shares the same x-axis
        ax2 = ax.twinx()
        
        # Plot log-scale data (clean and transformed) on main axis
        log_data = plot_df[
            (plot_df['Feature'] == feature) & 
            (plot_df['Dataset'].isin(['clean', 'transformed']))
        ]
        if not log_data.empty:
            # Violin plot with high transparency
            sns.violinplot(data=log_data,
                         x='Dataset',
                         y='Value',
                         hue='Well',
                         ax=ax,
                         inner=None, 
                         cut=0,
                         scale='width',
                         split=False,
                         alpha=0.5)  
            
            ax.set_yscale('log')
        
        # Plot linear-scale data (standardized) on second axis
        linear_data = plot_df[
            (plot_df['Feature'] == feature) & 
            (plot_df['Dataset'] == 'standardized')
        ]
        if not linear_data.empty:
            # Violin plot with high transparency
            sns.violinplot(data=linear_data,
                         x='Dataset',
                         y='Value',
                         hue='Well',
                         ax=ax2,
                         inner=None,  
                         cut=0,
                         scale='width',
                         split=False,
                         alpha=0.5)  
        
        # Customize plot
        ax.set_title(feature.replace('_', ' ').title())
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # Set y-axis labels
        ax.set_ylabel('Value (log scale)')
        ax2.set_ylabel('Value (linear scale)')
        
        # Remove redundant legends from violin plots
        if ax.get_legend():
            ax.get_legend().remove()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def test_missing_values(df, name):
    """
    Test for missing values in a dataframe, including NA, null, blank, and infinite values.
    Returns results in a format suitable for CSV export.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to test
    name : str
        Name of the dataset for printing
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing detailed missing value information
    """
    # Check for various types of missing or problematic values
    missing = pd.DataFrame({
        'null_na': df.isna().sum(),  # Catches np.nan, None, pd.NA
        'empty_string': df.astype(str).eq('').sum(),  # Empty strings
        'infinite': df.isin([np.inf, -np.inf]).sum()  # Infinite values
    })
    
    # Sum up all types of problematic values
    missing['total_issues'] = missing.sum(axis=1)
    
    # Calculate percentages
    missing['percentage'] = (missing['total_issues'] / len(df) * 100).round(2)
    
    # Add dataset name
    missing['dataset'] = name
    
    # Add column names as a separate column
    missing['column'] = missing.index
    
    # Reorder columns
    missing = missing[[
        'dataset', 'column', 'total_issues', 'percentage', 
        'null_na', 'empty_string', 'infinite'
    ]]
    
    # Filter for columns that have any issues
    results = missing[missing['total_issues'] > 0].reset_index(drop=True)
    
    # Print summary
    if not results.empty:
        print(f"\nMissing values in {name} dataset:")
        for _, row in results.iterrows():
            details = []
            if row['null_na'] > 0:
                details.append(f"{row['null_na']} NA/null")
            if row['empty_string'] > 0:
                details.append(f"{row['empty_string']} empty")
            if row['infinite'] > 0:
                details.append(f"{row['infinite']} infinite")
                
            print(f"{row['column']}: {row['total_issues']} total issues "
                  f"({row['percentage']:.2f}%) - {', '.join(details)}")
    else:
        print(f"\nNo missing values found in {name} dataset")
        
    return results
