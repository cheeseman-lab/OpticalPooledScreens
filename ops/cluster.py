"""
Clustering Algorithms and Utilities
This module provides a collection of clustering algorithms and related utilities
(relating to step 5 -- clustering). It includes functions for:

1. Data Loading and Processing: Loading and preprocessing gene expression data.
2. Feature Processing: Rank transformation, normalization, and feature selection.
3. Dimensionality Reduction: PCA and PHATE implementations.
4. Clustering: Leiden algorithm and PHATE-Leiden pipeline.
5. Cluster Analysis: Differential analysis and gene enrichment.
6. Database Integration: UniProt, STRING, and CORUM data processing.
7. Statistical Analysis: Correlation, testing, and effect size calculations.

"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import igraph
from igraph import Graph
import leidenalg
import phate
import os
import io
import gzip
import re
from itertools import combinations
import requests
from requests.adapters import HTTPAdapter, Retry

def load_gene_level_data(mitotic_path, interphase_path, all_path):
    """
    Load the three main dataframes and perform basic validation and cleaning
    
    Parameters:
    -----------
    mitotic_path : str
        Path to mitotic cells data
    interphase_path : str
        Path to interphase cells data
    all_path : str
        Path to combined data
        
    Returns:
    --------
    tuple of DataFrames
        (df_mitotic, df_interphase, df_all)
    """
    # Load dataframes
    df_mitotic = pd.read_csv(mitotic_path)
    df_interphase = pd.read_csv(interphase_path)
    df_all = pd.read_csv(all_path)
    
    # Function to clean and validate each dataframe
    def clean_and_validate_df(df, name):
        # Remove unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            
        # Check for required columns
        required_cols = ['gene_symbol_0', 'gene_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {name} dataset")
        
        # Reorder columns to ensure gene_count is after gene_symbol_0
        other_cols = [col for col in df.columns if col not in required_cols]
        new_cols = ['gene_symbol_0', 'gene_count'] + other_cols
        df = df[new_cols]
        df = df.rename(columns={'gene_count': 'cell_number'})

        return df
    
    # Clean and validate each dataframe
    df_mitotic = clean_and_validate_df(df_mitotic, 'mitotic')
    df_interphase = clean_and_validate_df(df_interphase, 'interphase')
    df_all = clean_and_validate_df(df_all, 'all')
    
    return df_mitotic, df_interphase, df_all

def calculate_mitotic_percentage(df_mitotic, df_interphase):
    """
    Calculate the percentage of mitotic cells for each gene using pre-grouped data,
    filling in zeros for missing genes in either dataset
    
    Parameters:
    -----------
    df_mitotic : DataFrame
        DataFrame containing mitotic cell data (already grouped by gene)
    df_interphase : DataFrame
        DataFrame containing interphase cell data (already grouped by gene)
        
    Returns:
    --------
    DataFrame
        Contains gene names and their mitotic percentages
    """
    # Get all unique genes from both datasets
    all_genes = sorted(list(set(df_mitotic['gene_symbol_0']) | set(df_interphase['gene_symbol_0'])))
    
    # Create dictionaries mapping genes to their counts
    mitotic_counts = dict(zip(df_mitotic['gene_symbol_0'], df_mitotic['cell_number']))
    interphase_counts = dict(zip(df_interphase['gene_symbol_0'], df_interphase['cell_number']))
    
    # Create result DataFrame with all genes, filling in zeros for missing values
    result_df = pd.DataFrame({
        'gene': all_genes,
        'mitotic_cells': [mitotic_counts.get(gene, 0) for gene in all_genes],
        'interphase_cells': [interphase_counts.get(gene, 0) for gene in all_genes]
    })
    
    # Report genes that were filled with zeros
    missing_in_mitotic = set(all_genes) - set(df_mitotic['gene_symbol_0'])
    missing_in_interphase = set(all_genes) - set(df_interphase['gene_symbol_0'])
    
    if missing_in_mitotic or missing_in_interphase:
        print("Note: Some genes were missing and filled with zero counts:")
        if missing_in_mitotic:
            print(f"Genes missing in mitotic data (filled with 0): {missing_in_mitotic}")
        if missing_in_interphase:
            print(f"Genes missing in interphase data (filled with 0): {missing_in_interphase}")
    
    # Calculate total cells and mitotic percentage
    result_df['total_cells'] = result_df['mitotic_cells'] + result_df['interphase_cells']
    
    # Handle division by zero: if total_cells is 0, set percentage to 0
    result_df['mitotic_percentage'] = np.where(
        result_df['total_cells'] > 0,
        (result_df['mitotic_cells'] / result_df['total_cells'] * 100).round(2),
        0.0
    )
    
    # Sort by mitotic percentage in descending order
    result_df = result_df.sort_values('mitotic_percentage', ascending=False)
    
    # Reset index to remove the old index
    result_df = result_df.reset_index(drop=True)
    
    # Print summary statistics
    print(f"\nProcessed {len(all_genes)} total genes")
    print(f"Average mitotic percentage: {result_df['mitotic_percentage'].mean():.2f}%")
    print(f"Median mitotic percentage: {result_df['mitotic_percentage'].median():.2f}%")
    
    return result_df

def split_channels(df, channel_pair, all_channels):
    """
    Filter dataframe to only include features from specified channel pair,
    removing features from other channels.
    
    Args:
        df (pd.DataFrame): Input dataframe with features
        channel_pair (str or tuple): Channels to keep. If 'all', keeps all channels
        all_channels (list): List of all possible channels
        
    Returns:
        pd.DataFrame: Filtered dataframe with features only from specified channels
    """
    # If 'all', return original dataset
    if channel_pair == 'all':
        return df.copy()
    
    # Find channels to remove (those not in channel_pair)
    channels_to_remove = [ch for ch in all_channels if ch not in channel_pair]
    
    # Get all column names
    columns = df.columns.tolist()
    
    # Find columns to remove (those containing removed channel names)
    columns_to_remove = [col for col in columns 
                        if any(ch in col for ch in channels_to_remove)]
    
    # Keep all columns except those from removed channels
    columns_to_keep = [col for col in columns if col not in columns_to_remove]
    
    return df[columns_to_keep]

def remove_low_number_genes(df, min_cells=10):
    """
    Remove genes with cell numbers below a certain threshold
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame containing 'cell_number' column
    min_cells : int, default=10
        Minimum number of cells required for a gene to be kept
        
    Returns:
    --------
    DataFrame
        DataFrame with genes filtered based on cell_number threshold
    """
    # Filter genes based on cell_number
    filtered_df = df[df['cell_number'] >= min_cells]
    
    # Print summary
    print("\nGene Filtering Summary:")
    print(f"Original genes: {len(df)}")
    print(f"Genes with < {min_cells} cells: {len(df) - len(filtered_df)}")
    print(f"Remaining genes: {len(filtered_df)}")
    
    return filtered_df

def remove_missing_features(df):
    """
    Remove features (columns) that contain any inf, nan, or blank values
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with features as columns
        
    Returns:
    --------
    DataFrame
        DataFrame with problematic features removed
    """
    import numpy as np
    
    df = df.copy()
    removed_features = {}
    
    # Check for infinite values
    inf_features = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if inf_features:
        removed_features['infinite'] = inf_features
        df = df.drop(columns=inf_features)
    
    # Check for null/na values
    null_features = df.columns[df.isna().any()].tolist()
    if null_features:
        removed_features['null_na'] = null_features
        df = df.drop(columns=null_features)
    
    # Check for empty strings (for string columns only)
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        empty_features = string_cols[df[string_cols].astype(str).eq('').any()].tolist()
        if empty_features:
            removed_features['empty_string'] = empty_features
            df = df.drop(columns=empty_features)
    
    # Print summary
    print("\nFeature Cleaning Summary:")
    print(f"Original features: {len(df.columns) + sum(len(v) for v in removed_features.values())}")
    
    if removed_features:
        print("\nRemoved features:")
        if 'infinite' in removed_features:
            print(f"\nFeatures with infinite values ({len(removed_features['infinite'])}):")
            for feat in removed_features['infinite']:
                print(f"- {feat}")
                
        if 'null_na' in removed_features:
            print(f"\nFeatures with null/NA values ({len(removed_features['null_na'])}):")
            for feat in removed_features['null_na']:
                print(f"- {feat}")
                
        if 'empty_string' in removed_features:
            print(f"\nFeatures with empty strings ({len(removed_features['empty_string'])}):")
            for feat in removed_features['empty_string']:
                print(f"- {feat}")
    else:
        print("\nNo problematic features found!")
    
    print(f"\nRemaining features: {len(df.columns)}")
    
    return df

def rank_transform(df, non_feature_cols=['gene_symbol_0']):
    """
    Transform features in a dataframe to their rank values, where highest value gets rank 1.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features to be ranked
    non_feature_cols : list
        List of column names that should not be ranked (e.g., identifiers, counts)
        
    Returns
    -------
    pd.DataFrame
        New dataframe with same structure but feature values replaced with ranks
    """
    # Get feature columns (all columns not in non_feature_cols)
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Create ranks for all feature columns at once
    ranked_features = df[feature_cols].rank(ascending=False).astype(int)
    
    # Combine non-feature columns with ranked features
    ranked = pd.concat([df[non_feature_cols], ranked_features], axis=1)
    
    return ranked

def select_features(df, correlation_threshold=0.9, variance_threshold=0.01, min_unique_values=5):
    """
    Select features based on correlation, variance, and unique values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features to be selected
    correlation_threshold : float, default=0.9
        Threshold for removing highly correlated features
    variance_threshold : float, default=0.01
        Threshold for removing low variance features
    min_unique_values : int, default=5
        Minimum unique values required for a feature to be kept

    Returns:
    --------
    tuple
        (DataFrame with selected features, dictionary of removed features)
    
    """
    import numpy as np
    import pandas as pd
    
    # Make a copy and handle initial column filtering
    df = df.copy()
    if 'cell_number' in df.columns:
        df = df.drop(columns=['cell_number'])
    
    # Store information about removed features
    removed_features = {
        'correlated': [],
        'low_variance': [],
        'few_unique_values': []
    }

    # Get numeric columns only, excluding gene_symbol_0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'gene_symbol_0']
    df_numeric = df[feature_cols]
    
    # Calculate correlation matrix once
    correlation_matrix = df_numeric.corr().abs()
    
    # Create a mask to get upper triangle of correlation matrix
    upper_tri = np.triu(np.ones(correlation_matrix.shape), k=1)
    high_corr_pairs = []
    
    # Get all highly correlated pairs at once
    pairs_idx = np.where((correlation_matrix.values * upper_tri) > correlation_threshold)
    for i, j in zip(*pairs_idx):
        high_corr_pairs.append((
            correlation_matrix.index[i],
            correlation_matrix.columns[j],
            correlation_matrix.iloc[i, j]
        ))
    
    # Process all correlated features at once
    if high_corr_pairs:
        # Calculate mean correlation for each feature
        mean_correlations = correlation_matrix.mean()
        
        # Track features to remove
        features_to_remove = set()
        
        # For each correlated pair, remove the feature with higher mean correlation
        for col1, col2, corr_value in high_corr_pairs:
            if col1 not in features_to_remove and col2 not in features_to_remove:
                feature_to_remove = col1 if mean_correlations[col1] > mean_correlations[col2] else col2
                features_to_remove.add(feature_to_remove)
                
                removed_features['correlated'].append({
                    'feature': feature_to_remove,
                    'correlated_with': col2 if feature_to_remove == col1 else col1,
                    'correlation': corr_value
                })
        
        df_numeric = df_numeric.drop(columns=list(features_to_remove))
    
    # Step 2: Remove low variance features (unchanged but done in one step)
    variances = df_numeric.var()
    low_variance_features = variances[variances < variance_threshold].index
    removed_features['low_variance'] = [
        {'feature': feat, 'variance': variances[feat]}
        for feat in low_variance_features
    ]
    df_numeric = df_numeric.drop(columns=low_variance_features)
    
    # Step 3: Remove features with few unique values (unchanged but done in one step)
    unique_counts = df_numeric.nunique()
    few_unique_features = unique_counts[unique_counts < min_unique_values].index
    removed_features['few_unique_values'] = [
        {'feature': feat, 'unique_values': unique_counts[feat]}
        for feat in few_unique_features
    ]
    df_numeric = df_numeric.drop(columns=few_unique_features)
    
    # Print summary
    print("\nFeature Selection Summary:")
    print(f"Original features: {len(numeric_cols)}")
    print(f"Features removed due to correlation: {len(removed_features['correlated'])}")
    print(f"Features removed due to low variance: {len(removed_features['low_variance'])}")
    print(f"Features removed due to few unique values: {len(removed_features['few_unique_values'])}")
    print(f"Final features: {len(df_numeric.columns)}")
    
    # Create final DataFrame with remaining numeric columns AND gene_symbol_0
    final_columns = ['gene_symbol_0'] + df_numeric.columns.tolist()
    
    return df[final_columns], removed_features

def normalize_to_controls(df, control_prefix='sg_nt'):
    """
    Normalize data using StandardScaler fit to control samples.
    Sets gene_symbol_0 as index if it isn't already.
    
    Args:
        df (pd.DataFrame): DataFrame to normalize
        control_prefix (str): Prefix identifying control samples in index or gene_symbol_0 column
        
    Returns:
        pd.DataFrame: Normalized DataFrame with gene symbols as index
    """
    df_copy = df.copy()
    
    # Handle cases where gene_symbol_0 might be a column or already the index
    if 'gene_symbol_0' in df_copy.columns:
        df_copy = df_copy.set_index('gene_symbol_0')
    
    # Fit scaler on control samples
    scaler = StandardScaler()
    control_mask = df_copy.index.str.startswith(control_prefix)
    scaler.fit(df_copy[control_mask].values)
    
    # Transform all data
    df_norm = pd.DataFrame(
        scaler.transform(df_copy.values),
        index=df_copy.index,
        columns=df_copy.columns
    )
    
    return df_norm

def perform_pca_analysis(df, variance_threshold=0.95, save_plot_path=None, random_state=42):
    """
    Perform PCA analysis and create explained variance plot.
    Expects gene_symbol_0 to be the index.
    
    Args:
        df (pd.DataFrame): Data with gene symbols as index
        variance_threshold (float): Cumulative variance threshold (default 0.95)
        save_plot_path (str): Path to save variance plot (optional)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (pca_df, n_components, pca_object)
            - pca_df: DataFrame with PCA transformed data (gene symbols as index)
            - n_components: Number of components needed to reach variance threshold
            - pca_object: Fitted PCA object
    """
    # Initialize and fit PCA
    pca = PCA(random_state=random_state)
    pca_transformed = pca.fit_transform(df)
    
    # Create DataFrame with PCA results
    n_components_total = pca_transformed.shape[1]
    pca_df = pd.DataFrame(
        pca_transformed,
        columns=[f'pca_{n}' for n in range(n_components_total)],
        index=df.index
    )
    
    # Find number of components needed for threshold
    cumsum = pca.explained_variance_ratio_.cumsum()
    n_components = np.argwhere(cumsum >= variance_threshold)[0][0] + 1
    
    # Create variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(cumsum, '-')
    plt.axhline(variance_threshold, linestyle='--', color='red', 
                label=f'{variance_threshold*100}% Threshold')
    plt.axvline(n_components, linestyle='--', color='blue', 
                label=f'n={n_components}')
    plt.ylabel('Cumulative fraction of variance explained')
    plt.xlabel('Number of principal components included')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    
    if save_plot_path:
        plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"Number of components needed for {variance_threshold*100}% variance: {n_components}")
    print(f"Shape of input data: {df.shape}")
    
    # Create threshold-limited version
    pca_df_threshold = pca_df[[f'pca_{i}' for i in range(n_components)]]
    
    print(f"Shape of PCA transformed and reduced data: {pca_df_threshold.shape}")

    return pca_df_threshold, n_components, pca

def run_phate(df, random_state=42, n_jobs=4, knn=10, metric='euclidean', **kwargs):
    """
    Run PHATE dimensionality reduction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data matrix
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=4
        Number of parallel jobs
    knn : int, default=10
        Number of nearest neighbors
    metric : str, default='euclidean'
        Distance metric for KNN
    **kwargs : dict
        Additional arguments passed to PHATE
        
    Returns:
    --------
    tuple
        (DataFrame with PHATE coordinates, PHATE object)
    """
    # Initialize and run PHATE
    p = phate.PHATE(
        random_state=random_state,
        n_jobs=n_jobs,
        knn=knn,
        knn_dist=metric,
        **kwargs
    )
    
    # Transform data
    X_phate = p.fit_transform(df.values)
    
    # Create output DataFrame
    df_phate = pd.DataFrame(
        X_phate,
        index=df.index,
        columns=['PHATE_0', 'PHATE_1']
    )
    
    return df_phate, p

def run_leiden_clustering(weights, resolution=1.0, seed=42):
    """
    Run Leiden clustering on a weighted adjacency matrix.
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Weighted adjacency matrix
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering
    seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    list
        Cluster assignments
    """
    # Force symmetry by averaging with transpose
    weights_symmetric = (weights + weights.T) / 2
    
    # Create graph from symmetrized weights
    g = Graph().Weighted_Adjacency(
        matrix=weights_symmetric.tolist(),
        mode='undirected'
    )
    
    # Run Leiden clustering
    partition = leidenalg.find_partition(
        g,
        partition_type=leidenalg.RBConfigurationVertexPartition,
        weights=g.es['weight'],
        n_iterations=-1,
        seed=seed,
        resolution_parameter=resolution
    )
    
    return partition.membership

def phate_leiden_pipeline(df, resolution=1.0, phate_kwargs=None):
    """
    Run complete PHATE and Leiden clustering pipeline.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data matrix
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering
    phate_kwargs : dict, optional
        Additional arguments for PHATE
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with PHATE coordinates and cluster assignments
    """
    # Default PHATE parameters
    if phate_kwargs is None:
        phate_kwargs = {}
    
    # Run PHATE
    df_phate, p = run_phate(df, **phate_kwargs)
    
    # Get weights from PHATE
    weights = np.asarray(p.graph.diff_op.todense())
    
    # Run Leiden clustering
    clusters = run_leiden_clustering(weights, resolution=resolution)
    
    # Add clusters to results
    df_phate['cluster'] = clusters

    # Sort by cluster
    df_phate = df_phate.sort_values('cluster')

    # Print number of clusters and average cluster size
    print(f"Number of clusters: {df_phate['cluster'].nunique()}")
    print(f"Average cluster size: {df_phate['cluster'].value_counts().mean():.2f}")
    
    return df_phate

def get_uniprot_data():
    """
    Fetch all human reviewed UniProt data using REST API

    Returns:
    --------
    pandas.DataFrame
        DataFrame with UniProt data
    """
    # Define UniProt REST API query
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    # Function to extract next link from headers
    def get_next_link(headers):
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)
    
    # Fetch UniProt data
    url = "https://rest.uniprot.org/uniprotkb/search"
    # Query for human reviewed entries with specific fields
    params = {
        'query': 'organism_id:9606 AND reviewed:true',
        'fields': 'gene_names,cc_function,xref_kegg,xref_complexportal,xref_string',
        'format': 'tsv',
        'size': 500
    }
    
    # Fetch data in batches
    initial_response = session.get(url, params=params)
    batch_url = initial_response.url
    results = []
    progress = 0
    
    # Process each batch
    print("Fetching UniProt data...")
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = int(response.headers["x-total-results"])
        
        lines = response.text.splitlines()
        if progress == 0:
            headers = lines[0].split('\t')
        
        for line in lines[1:] if progress == 0 else lines:
            results.append(line.split('\t'))
        
        progress += len(lines[1:] if progress == 0 else lines)
        print(f'Progress: {progress} / {total}')
        
        batch_url = get_next_link(response.headers)
    
    # Create DataFrame from results
    df = pd.DataFrame(results, columns=headers)
    print(f"Completed. Total entries: {len(df)}")
    return df

def merge_phate_uniprot(df_phate, database_path="databases"):
    """
    Merge PHATE clustering results with UniProt data

    Parameters:
    -----------
    df_phate : pandas.DataFrame
        DataFrame with PHATE coordinates and cluster assignments
    database_path : str, default='databases'
        Path to saving/loading UniProt data

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with UniProt data
    
    """
    # Make a copy to avoid modifying the original
    df_phate = df_phate.copy()
    
    # If gene_symbol_0 is in the index, reset it to become a column
    if df_phate.index.name == 'gene_symbol_0':
        df_phate = df_phate.reset_index()
    # If we still don't have gene_symbol_0 as a column, create it from the index
    elif 'gene_symbol_0' not in df_phate.columns:
        df_phate['gene_symbol_0'] = df_phate.index
        df_phate = df_phate.reset_index(drop=True)
    
    uniprot_file = os.path.join(database_path, "uniprot_complete_data.csv")
    
    # Load UniProt data
    if not os.path.exists(uniprot_file):
        uniprot_df = get_uniprot_data()
        uniprot_df.to_csv(uniprot_file, index=False)
        print(f"Saved {len(uniprot_df)} UniProt entries to {uniprot_file}")
    else:
        uniprot_df = pd.read_csv(uniprot_file)
    
    # Split gene names and explode
    uniprot_df['gene_names'] = uniprot_df['Gene Names'].str.split()
    uniprot_df = uniprot_df.explode('gene_names')
    uniprot_df.rename(columns={'Function [CC]': 'Function'}, inplace=True)
    
    # Merge with PHATE data
    result = pd.merge(
        df_phate,
        uniprot_df.rename(columns={'gene_names': 'gene_symbol_0'}),
        on='gene_symbol_0',
        how='left'
    )

    # Remove duplicate columns
    for col in result.columns:
        if result[col].dtype == 'object':
            result[col] = result[col].str.replace(';', '')
    
    return result

def create_cluster_gene_table(df, cluster_col='cluster', columns_to_combine=['gene_symbol_0']):
    """
    Creates a table with cluster number and combined gene information.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cluster assignments and gene information
    cluster_col : str, default='cluster'
    columns_to_combine : list, default=['gene_symbol_0']
        Columns to combine for each cluster

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster number, combined gene information, and gene count

    """
    # Combine gene information for each cluster
    cluster_summary = df.groupby(cluster_col).agg({
        col: lambda x: ', '.join(sorted([str(val) for val in set(x) if pd.notna(val)]))
        for col in columns_to_combine
    }).reset_index()
    
    # Count number of unique genes in each cluster
    cluster_summary['gene_number'] = df.groupby(cluster_col)[columns_to_combine[0]].agg(
        lambda x: len([val for val in set(x) if pd.notna(val)])
    )
    
    # Sort by cluster number
    cluster_summary = cluster_summary.rename(columns={cluster_col: 'cluster_number'})
    cluster_summary = cluster_summary.sort_values('cluster_number').reset_index(drop=True)
    
    return cluster_summary

def analyze_differential_features(cluster_gene_table, feature_df, n_top=5, exclude_cols=['gene_symbol_0', 'cell_number']):
    """
    Analyze differential features between clusters

    Parameters:
    -----------
    cluster_gene_table : pandas.DataFrame
        DataFrame with cluster assignments and gene information
    feature_df : pandas.DataFrame
        DataFrame with feature values for each gene
    n_top : int, default=5
        Number of top features to select
    exclude_cols : list, default=['gene_symbol_0', 'cell_number']
        Columns to exclude from feature analysis

    Returns:
    --------
    tuple
        (DataFrame with top features for each cluster, dictionary of feature analysis results)   

    """
    # Get feature columns
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    results = {}
    
    # Copy the cluster gene table
    cluster_gene_table = cluster_gene_table.copy()
    cluster_gene_table[f'top_{n_top}_up'] = ''
    cluster_gene_table[f'top_{n_top}_down'] = ''
    
    # Analyze each cluster
    total_clusters = len(cluster_gene_table)
    print(f"Analyzing {total_clusters} clusters...")
    
    # Iterate over each cluster
    for idx, row in enumerate(cluster_gene_table.iterrows(), 1):
        cluster_num = row[1]['cluster_number']
        cluster_genes = set(row[1]['gene_symbol_0'].split(', '))
        
        print(f"Processing cluster {idx}/{total_clusters} (#{cluster_num})", end='\r')
        
        # Split data into cluster and non-cluster
        cluster_data = feature_df[feature_df['gene_symbol_0'].isin(cluster_genes)][feature_cols]
        non_cluster_data = feature_df[~feature_df['gene_symbol_0'].isin(cluster_genes)][feature_cols]
        
        # Perform t-test for each feature
        t_stats, p_values, effect_sizes = [], [], []
        
        # Calculate t-statistic, p-value, and effect size for each feature
        for feature in feature_cols:
            t_stat, p_val = stats.ttest_ind(cluster_data[feature], non_cluster_data[feature])
            cohens_d = (cluster_data[feature].mean() - non_cluster_data[feature].mean()) / np.sqrt(
                ((cluster_data[feature].std() ** 2 + non_cluster_data[feature].std() ** 2) / 2))
            
            # Store results
            t_stats.append(abs(t_stat))
            p_values.append(p_val)
            effect_sizes.append(cohens_d)
            
        # Store results in DataFrame
        feature_results = pd.DataFrame({
            'feature': feature_cols,
            't_statistic': t_stats,
            'p_value': p_values,
            'effect_size': effect_sizes
        })
        
        # Adjust p-values using Benjamini-Hochberg method
        feature_results['p_value_adj'] = multipletests(feature_results['p_value'], method='fdr_bh')[1]
        feature_results['abs_effect_size'] = feature_results['effect_size'].abs()
        
        # Select top features based on effect size
        top_up = feature_results.nlargest(n_top, 'effect_size')
        top_down = feature_results.nsmallest(n_top, 'effect_size')
        
        # Update cluster gene table with top features
        cluster_idx = cluster_gene_table.index[cluster_gene_table['cluster_number'] == cluster_num][0]
        cluster_gene_table.at[cluster_idx, f'top_{n_top}_up'] = ', '.join(top_up['feature'])
        cluster_gene_table.at[cluster_idx, f'top_{n_top}_down'] = ', '.join(top_down['feature'])
        
        # Store feature analysis results
        results[cluster_num] = feature_results
        
    return cluster_gene_table, results

def get_string_data():
    """
    Fetch STRING interaction data for human proteins
    """
    print("Fetching STRING data...")
    url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    
    response = requests.get(url)
    response.raise_for_status()
    
    # Read compressed data directly into DataFrame
    with gzip.open(io.BytesIO(response.content), 'rt') as f:
        df = pd.read_csv(f, sep=' ')
    
    # Filter interactions with combined score >= 950
    df = df[df['combined_score'] >= 950]
    print(f"Completed. Total interactions: {len(df)}")
    return df

def get_corum_data():
    """
    Fetch CORUM complex data for human proteins
    """
    print("Fetching CORUM data...")
    url = "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_current_file"
    
    # Parameters for human complexes in text format
    params = {
        "file_id": "human",
        "file_format": "txt"
    }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    # Read data into DataFrame
    df = pd.read_csv(io.StringIO(response.text), sep='\t')
    print(f"Completed. Total complexes: {len(df)}")
    return df

def process_interactions(df_clusters, database_path="databases"):
    """
    Process cluster data against STRING and CORUM databases

    Parameters:
    -----------
    df_clusters : pandas.DataFrame
        DataFrame with cluster information
    database_path : str, default='databases'
        Path to saving/loading STRING and CORUM data

    Returns:
    --------
    tuple:
        - DataFrame with cluster information and validation results
        - Dictionary with global metrics for both STRING and CORUM
    """
    # Define file paths using os.path.join
    string_file = os.path.join(database_path, "9606.protein.links.v12.0.txt")
    corum_file = os.path.join(database_path, "corum_humanComplexes.txt")
    
    # Load STRING data
    if not os.path.exists(string_file):
        string_df = get_string_data()
        string_df.to_csv(string_file, sep='\t', index=False)
        print(f"Saved {len(string_df)} STRING interactions to {string_file}")
    else:
        string_df = pd.read_csv(string_file, sep='\t')
        
    # Load CORUM data
    if not os.path.exists(corum_file):
        corum_df = get_corum_data()
        corum_df.to_csv(corum_file, sep='\t', index=False)
        print(f"Saved {len(corum_df)} CORUM complexes to {corum_file}")
    else:
        corum_df = pd.read_csv(corum_file, sep='\t')
    
    # Process STRING data
    string_pairs = set(map(tuple, string_df[['protein1', 'protein2']].values))
    
    # Process CORUM data - keep both pair and complex information
    corum_complexes = []
    corum_pairs = set()
    for _, complex_row in corum_df.iterrows():
        if pd.isna(complex_row['subunits_gene_name']):
            continue
        # Get all genes in complex
        genes = [gene.strip() for gene in complex_row['subunits_gene_name'].split(';')]
        if len(genes) >= 2:
            # Store complete complex
            corum_complexes.append({
                'name': complex_row['complex_name'],
                'genes': set(genes),
                'size': len(genes)
            })
            # Store pairs for pair-based analysis
            pairs = set(combinations(sorted(genes), 2))
            corum_pairs.update(pairs)
    
    # Get all screened genes (from both STRING and CORUM columns)
    screened_genes = set()
    for _, row in df_clusters.iterrows():
        screened_genes.update(gene.strip() for gene in row['gene_symbol_0'].replace(', ', ',').split(','))
        screened_genes.update(gene.strip() for gene in row['STRING'].replace(', ', ',').split(','))
    
    # Process cluster data
    all_string_predicted_pairs = set()
    all_cluster_pairs = set()
    all_corum_cluster_pairs = set()
    results = []
    
    for _, row in df_clusters.iterrows():
        cluster_num = row['cluster_number']
        cluster_genes = set(gene.strip() for gene in row['gene_symbol_0'].replace(', ', ',').split(','))
        genes_string = set(gene.strip() for gene in row['STRING'].replace(', ', ',').split(','))
        
        # STRING analysis
        string_cluster_pairs = set()
        if len(genes_string) >= 2:
            string_cluster_pairs = set(combinations(sorted(genes_string), 2))
            all_string_predicted_pairs.update(string_cluster_pairs)
        matching_string_pairs = string_cluster_pairs & string_pairs if string_cluster_pairs else set()
        
        # CORUM complex-level analysis
        enriched_complexes = []
        for complex_info in corum_complexes:
            complex_genes = complex_info['genes']
            screened_complex_genes = complex_genes & screened_genes
            
            # Apply complex filtering criteria
            if (len(screened_complex_genes) >= 3 and 
                len(screened_complex_genes) >= (2/3 * len(complex_genes))):
                
                # Calculate overlap with cluster
                overlap_genes = cluster_genes & screened_complex_genes
                
                # Fisher's exact test
                table = [
                    [len(overlap_genes), len(screened_complex_genes - cluster_genes)],
                    [len(cluster_genes - screened_complex_genes), 
                     len(screened_genes - cluster_genes - screened_complex_genes)]
                ]
                odds_ratio, pvalue = fisher_exact(table)
                
                if pvalue < 0.05:  # Store for FDR correction
                    enriched_complexes.append({
                        'complex_name': complex_info['name'],
                        'pvalue': pvalue,
                        'overlap_size': len(overlap_genes),
                        'complex_size': len(screened_complex_genes)
                    })
        
        # Apply FDR correction to enriched complexes
        if enriched_complexes:
            pvals = [x['pvalue'] for x in enriched_complexes]
            _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
            significant_complexes = [
                enr for enr, p_adj in zip(enriched_complexes, pvals_corrected)
                if p_adj < 0.05
            ]
        else:
            significant_complexes = []
            
        # CORUM pair-based analysis for this cluster
        if len(cluster_genes) >= 2:
            cluster_pairs = set(combinations(sorted(cluster_genes), 2))
            all_cluster_pairs.update(cluster_pairs)
            # Find which pairs are also in CORUM
            matching_corum = cluster_pairs & corum_pairs
            all_corum_cluster_pairs.update(matching_corum)
        
        # Store results
        results.append({
            'cluster_number': cluster_num,
            'total_string_pairs': len(string_cluster_pairs),
            'string_validated_pairs': len(matching_string_pairs),
            'string_validation_ratio': len(matching_string_pairs) / len(string_cluster_pairs) if string_cluster_pairs else 0,
            'enriched_corum_complexes': ', '.join(x['complex_name'] for x in significant_complexes),
            'num_enriched_complexes': len(significant_complexes)
        })
    
    # Calculate global STRING metrics
    string_true_positives = all_string_predicted_pairs & string_pairs
    string_precision = len(string_true_positives) / len(all_string_predicted_pairs) if all_string_predicted_pairs else 0
    string_recall = len(string_true_positives) / len(string_pairs) if string_pairs else 0
    string_f1 = 2 * (string_precision * string_recall) / (string_precision + string_recall) if (string_precision + string_recall) else 0
    
    # Calculate CORUM pair-based metrics (as shown in the plot)
    corum_precision = len(all_corum_cluster_pairs) / len(all_cluster_pairs) if all_cluster_pairs else 0
    corum_recall = len(all_corum_cluster_pairs) / len(corum_pairs) if corum_pairs else 0
    corum_f1 = 2 * (corum_precision * corum_recall) / (corum_precision + corum_recall) if (corum_precision + corum_recall) else 0
    
    results_df = pd.DataFrame(results)
    cluster_results = df_clusters.merge(results_df, on='cluster_number')
    
    # Global metrics including both STRING and CORUM results
    global_metrics = {
        'num_clusters': df_clusters['cluster_number'].nunique(),  
        'string_global_precision': string_precision,
        'string_global_recall': string_recall,
        'string_global_f1': string_f1,
        'string_total_predicted_pairs': len(all_string_predicted_pairs),
        'string_total_reference_pairs': len(string_pairs),
        'string_total_correct_pairs': len(string_true_positives),
        'corum_precision': corum_precision,  # Fraction of cluster pairs in CORUM
        'corum_recall': corum_recall,        # Fraction of CORUM pairs in clusters
        'corum_f1': corum_f1,
        'corum_total_cluster_pairs': len(all_cluster_pairs),
        'corum_total_complex_pairs': len(corum_pairs),
        'corum_matching_pairs': len(all_corum_cluster_pairs),
        'num_enriched_complexes': sum(len(row['enriched_corum_complexes']) for row in results)
    }
    
    return cluster_results, global_metrics

def aggregate_resolution_metrics(output_dir, dataset_types, channel_pairs, leiden_resolutions):
    """
    Aggregate metrics across different resolutions into a single CSV.

    Parameters:
    -----------
    output_dir : str
        Directory containing resolution-wise metrics
    dataset_types : list
        List of dataset types to include
    channel_pairs : list
        List of channel pairs to include
    leiden_resolutions : list
        List of Leiden resolutions to include

    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated metrics
    
    """
    all_metrics = []
    
    for channel_pair in channel_pairs:
        # Handle channel pair directory name
        if channel_pair == 'all':
            pair_dir = os.path.join(output_dir, "all_channels")
        else:
            pair_dir = os.path.join(output_dir, f"channels_{'_'.join(channel_pair)}")
            
        for dataset_type in dataset_types:
            for resolution in leiden_resolutions:
                resolution_dir = os.path.join(pair_dir, f"resolution_{resolution}")
                metrics_file = os.path.join(resolution_dir, "csv", f"{dataset_type}_global_metrics.txt")
                
                if os.path.exists(metrics_file):
                    # Read metrics
                    metrics = {}
                    with open(metrics_file, 'r') as f:
                        for line in f:
                            key, value = line.strip().split(': ')
                            metrics[key] = float(value)
                    
                    # Add metadata
                    metrics['resolution'] = resolution
                    metrics['dataset_type'] = dataset_type
                    metrics['channel_pair'] = str(channel_pair)
                    
                    all_metrics.append(metrics)
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(all_metrics)
        
    return df_metrics