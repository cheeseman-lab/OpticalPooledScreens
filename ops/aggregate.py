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
import matplotlib.pyplot as plt

def load_hdf_subset(file_path, fraction=0.01, seed=42):
    """
    Load a random fraction of rows from an HDF file.
    
    Parameters
    ----------
    file_path : str
        Path to HDF file
    fraction : float
        Fraction of rows to load (between 0 and 1)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Random subset of the data
    """
    # First try to get the total number of rows
    full_df = pd.read_hdf(file_path)
    nrows = len(full_df)
    
    # Calculate number of rows to keep
    np.random.seed(seed)
    n_samples = int(nrows * fraction)
    
    # Randomly select rows
    selected_idx = np.random.choice(nrows, size=n_samples, replace=False)
    df = full_df.iloc[selected_idx]
    
    print(f"Loaded {len(df):,} cells ({fraction*100:.1f}% of {nrows:,} total cells)")
    
    return df

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
            
        return df_out
    else:
        raise ValueError("Only method='median' is currently supported")

def collapse_to_gene(df, target_features=None, index_features=['gene_symbol_0'], 
                    min_count=None):
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

def plot_mitotic_distribution(df, threshold_variable, threshold_value, range=(-5, 20), bins=100):
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
    range : tuple
        (min, max) values for plotting histogram
    bins : int
        Number of bins for histogram
        
    Returns
    -------
    float
        Percentage of cells classified as mitotic
    """
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(df[threshold_variable], bins=bins, range=range)
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