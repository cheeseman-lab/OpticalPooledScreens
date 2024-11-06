"""
Clustering Algorithms and Utilities
This module provides a collection of clustering algorithms and related utilities
(relating to step 5 -- clustering). It includes functions for:

1. Loading and preprocessing data for clustering.
2. Affinity-based Clustering: Implementation of the Leiden algorithm for community detection.
3. Hierarchical Clustering: Functions for hierarchical clustering and dendrogram manipulation.
4. Density-based Clustering: DBSCAN and HDBSCAN implementations.
5. Spectral Clustering: Implementation of spectral clustering algorithm.
6. Consensus Clustering: Methods for combining multiple clustering results.
7. Cluster Refinement: Utilities for merging small clusters and improving clustering results.

"""

import leidenalg
import graphtools
import hdbscan
import sklearn
from sklearn import cluster
from igraph import Graph
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import phate
import leidenalg

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

# REMOVE ALL BELOW?

class AffinityLeiden(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    Affinity Leiden clustering algorithm.

    This class implements a clustering algorithm using an affinity matrix and the Leiden algorithm for community detection.
    The algorithm constructs an affinity matrix from input data, then applies the Leiden algorithm to detect clusters.

    Attributes:
        knn (int): Number of nearest neighbors for constructing the graph.
        knn_max (int, optional): Maximum number of nearest neighbors to consider.
        knn_dist (str): Distance metric for nearest neighbor search (default is "euclidean").
        n_pca (int): Number of principal components to use for dimensionality reduction.
        decay (float): Decay parameter for the graph construction.
        n_landmark (int): Number of landmarks for the graph construction.
        resolution_parameter (float): Resolution parameter for the Leiden algorithm.
        n_jobs (int): Number of parallel jobs to run (default is 1).
        verbose (bool): If True, print progress information.
        random_state (int, optional): Seed for the random number generator.
    """

    def __init__(
        self,
        knn=5,
        knn_max=None,
        knn_dist="euclidean",
        n_pca=100,
        decay=40,
        n_landmark=2000,
        resolution_parameter=1,
        n_jobs=1,
        verbose=True,
        random_state=None,
    ):
        """
        Initialize the AffinityLeiden clustering algorithm.

        Parameters:
            knn (int): Number of nearest neighbors for graph construction.
            knn_max (int, optional): Maximum number of nearest neighbors.
            knn_dist (str): Distance metric for nearest neighbor search.
            n_pca (int): Number of principal components.
            decay (float): Decay parameter for the graph.
            n_landmark (int): Number of landmarks.
            resolution_parameter (float): Resolution parameter for Leiden algorithm.
            n_jobs (int): Number of parallel jobs.
            verbose (bool): If True, print progress information.
            random_state (int, optional): Random seed for reproducibility.
        """
        self.knn = knn
        self.knn_max = knn_max
        self.knn_dist = knn_dist
        self.decay = decay
        self.n_pca = n_pca
        self.n_jobs = n_jobs
        self.n_landmark = n_landmark
        self.resolution_parameter = resolution_parameter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the AffinityLeiden model to the data.

        Parameters:
            X (array-like or list): Input data.
            y (array-like, optional): Target values (not used).

        Returns:
            self: The fitted instance of the AffinityLeiden class.
        """
        if isinstance(X, list):
            X = np.array(X)

        if X.ndim < 2:
            raise ValueError("Cannot fit 1D array.")

        if X.shape[0] == 1:
            raise ValueError("Input contains only 1 sample.")

        self.n_features_in_ = X.shape[1]

        # Create the graph from input data
        graph = graphtools.Graph(
            X,
            n_pca=self.n_pca,
            n_landmark=self.n_landmark,
            distance=self.knn_dist,
            knn=self.knn,
            knn_max=self.knn_max,
            decay=self.decay,
            thresh=1e-4,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        # Compute the affinity matrix
        self.affinity_matrix_ = graph.diff_op.toarray()

        # Create a weighted adjacency graph for Leiden algorithm
        affinity_igraph = Graph().Weighted_Adjacency(
            matrix=self.affinity_matrix_.tolist(), mode="undirected"
        )

        # Apply the Leiden algorithm for community detection
        partition = leidenalg.find_partition(
            affinity_igraph,
            partition_type=leidenalg.RBConfigurationVertexPartition,
            weights=affinity_igraph.es["weight"],
            n_iterations=-1,
            seed=self.random_state,
            resolution_parameter=self.resolution_parameter,
        )

        self.labels_ = np.array(partition.membership)
        self.q_ = partition.q
        return self

    @property
    def singularities(self):
        """
        Number of singleton clusters (clusters with only one member).

        Returns:
            int: Number of singleton clusters.
        """
        self._singularities = (
            np.unique(self.labels_, return_counts=True)[1] == 1
        ).sum()
        return self._singularities

    @property
    def n_clusters(self):
        """
        Number of unique clusters identified by the algorithm.

        Returns:
            int: Number of clusters.
        """
        self._n_labels = len(np.unique(self.labels_))
        return self._n_labels

    @property
    def mean_cluster_size(self):
        """
        Mean size of clusters.

        Returns:
            float: Mean size of the clusters.
        """
        self._mean_cluster_size = np.unique(self.labels_, return_counts=True)[1].mean()
        return self._mean_cluster_size

    def set_index(self, index):
        """
        Set custom index for the data.

        Parameters:
            index (array-like): Custom index.

        Returns:
            self: The instance with the custom index set.
        """
        assert len(index) == len(self.labels_)
        self.index = index
        return self

    def adjusted_mutual_info_score(self, s):
        """
        Compute the adjusted mutual information score between true labels and provided labels.

        Parameters:
            s (DataFrame): DataFrame with true labels.

        Returns:
            float: Adjusted mutual information score.
        """
        return sklearn.metrics.adjusted_mutual_info_score(
            self.labels_, s[self.index].values
        )

    def adjusted_rand_score(self, s):
        """
        Compute the adjusted Rand index between true labels and provided labels.

        Parameters:
            s (DataFrame): DataFrame with true labels.

        Returns:
            float: Adjusted Rand index.
        """
        return sklearn.metrics.adjusted_rand_score(self.labels_, s[self.index].values)

def fcluster_to_df(Z, n_clusters, index):
    """
    Convert hierarchical clustering linkage matrix to a DataFrame with cluster labels.

    Parameters:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        n_clusters (int): The number of clusters to form.
        index (Index): Index for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    clusters = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def leiden_to_df(adjacency, resolution, index, seed=42):
    """
    Perform Leiden clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        resolution (float): Resolution parameter for Leiden algorithm.
        index (Index): Index for the DataFrame.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    igraph = Graph().Weighted_Adjacency(matrix=adjacency, mode="undirected")

    partition = leidenalg.find_partition(
        igraph,
        partition_type=leidenalg.RBConfigurationVertexPartition,
        weights=igraph.es["weight"],
        n_iterations=-1,
        seed=seed,
        resolution_parameter=resolution,
    )

    return pd.DataFrame(partition.membership, index=index, columns=['cluster'])

def dbscan_to_df(adjacency, eps, min_samples, index):
    """
    Perform DBSCAN clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        index (Index): Index for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    distance = 1 - (adjacency / adjacency.max())
    clusters = cluster.dbscan(distance, metric='precomputed', eps=eps, min_samples=min_samples)[1]
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def hdbscan_to_df(adjacency, min_cluster_size, min_samples, index):
    """
    Perform HDBSCAN clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        min_cluster_size (int): The minimum size of a cluster.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        index (Index): Index for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    distance = 1 - (adjacency / adjacency.max())
    clusters = hdbscan.hdbscan(distance, metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)[0]
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def spectral_to_df(adjacency, n_clusters, index, seed=42):
    """
    Perform spectral clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        n_clusters (int): The number of clusters to form.
        index (Index): Index for the DataFrame.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    if not np.array_equal(adjacency, adjacency.T):
        adjacency = (adjacency + adjacency.T) / 2

    clusters = cluster.spectral_clustering(adjacency, n_clusters=n_clusters, random_state=seed)
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def consensus_matrix(dfs, column="cluster", weights=None, combine_nt=False, nt_threshold=1):
    """
    Compute the consensus matrix from multiple clustering DataFrames.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames with cluster labels.
        column (str): Column name for cluster labels in each DataFrame.
        weights (list of float, optional): Weights for each DataFrame. Must sum to 1.
        combine_nt (bool, optional): Whether to combine non-target clusters.
        nt_threshold (int, optional): Threshold for combining non-target clusters.

    Returns:
        pd.DataFrame: Consensus matrix.
    """
    # Check for identical indices across DataFrames
    [pd.testing.assert_index_equal(dfs[0].index, df.index) for df in dfs[1:]]

    if weights is not None:
        if sum(weights) != 1:
            raise ValueError("`weights` must sum to 1")
        if len(weights) != len(dfs):
            raise ValueError("`weights` must have the same length as `dfs`")
    else:
        weights = [1 / len(dfs)] * len(dfs)

    C = np.zeros((dfs[0].pipe(len),) * 2)

    for df, w in zip(dfs, weights):
        if combine_nt:
            df = combine_nt_clusters(df, col=column, nt_threshold=nt_threshold)
        C += np.equal(*np.meshgrid(*(df[column].values,) * 2)) * w

    df_consensus = pd.DataFrame(C, columns=dfs[0].index, index=dfs[0].index)
    return df_consensus

def subsampled_consensus_matrix(dfs, column="cluster", combine_nt=False, nt_threshold=1, tqdm=False):
    """
    Compute the subsampled consensus matrix from multiple clustering DataFrames.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames with cluster labels.
        column (str): Column name for cluster labels in each DataFrame.
        combine_nt (bool, optional): Whether to combine non-target clusters.
        nt_threshold (int, optional): Threshold for combining non-target clusters.
        tqdm (bool, optional): Whether to display a progress bar.

    Returns:
        pd.DataFrame: Subsampled consensus matrix.
    """
    full_index = set()
    for df in dfs:
        full_index |= set(df.index)

    full_index = pd.MultiIndex.from_tuples(sorted(full_index), names=["gene_symbol", "gene_id"])

    C = np.zeros((len(full_index),) * 2)
    C_count = np.zeros_like(C)

    if tqdm:
        from tqdm.auto import tqdm
        dfs = tqdm(dfs)

    for df in dfs:
        df = df.sort_index()
        if combine_nt:
            df = combine_nt_clusters(df, col=column, nt_threshold=nt_threshold)
        selected = np.argwhere(full_index.isin(df.index))
        C_count[selected, selected.T] += 1
        C[selected, selected.T] += np.equal(*np.meshgrid(*(df[column].values,) * 2)).astype(int)

    df_consensus = pd.DataFrame(C / C_count, columns=full_index, index=full_index)
    return df_consensus

def linkage_clustermap(df_similarity, method="single"):
    """
    Create a clustermap based on a similarity matrix using hierarchical clustering.

    Parameters:
        df_similarity (pd.DataFrame): Similarity matrix.
        method (str): Method for hierarchical clustering (e.g., "single", "complete").

    Returns:
        sns.matrix.ClusterGrid: Clustermap object.
    """
    linkage = getattr(hierarchy, method)(squareform(1 - df_similarity.values))
    cm = sns.clustermap(df_similarity, row_linkage=linkage, col_linkage=linkage)
    return cm

def combine_nt_clusters(df, col="cluster", nt_threshold=1):
    """
    Combine non-target clusters in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with cluster labels.
        col (str): Column name for cluster labels.
        nt_threshold (int): Threshold for combining non-target clusters.

    Returns:
        pd.DataFrame: DataFrame with non-target clusters combined.
    """
    df_ = df.copy()
    if nt_threshold is None:
        df_ = df_.astype({col: "category"})
        s = (
            df_.query('gene_id=="-1"')[col].value_counts(normalize=True).sort_index()
            > df_[col].value_counts(normalize=True).sort_index()
        )
        nt_clusters = list(s.index.categories[s])
    else:
        nt_clusters = list(
            df_.query('gene_id=="-1"')[col]
            .value_counts()
            .rename("nt_count")
            .to_frame()
            .query("nt_count>=@nt_threshold")
            .index
        )
    df_ = df_.astype({col: int})
    df_.loc[df_[col].isin(nt_clusters), col] = -1
    return df_

def fcluster_combine_leaves(Z, t, criterion="distance", depth=2, R=None, monocrit=None):
    """
    Combine clusters in a hierarchical clustering linkage matrix to ensure no leaf clusters remain.

    Parameters:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        t (float): The threshold to apply.
        criterion (str): Criterion to use for cluster formation (default is "distance").
        depth (int): The depth of the tree to use for cluster formation.
        R (float, optional): Criterion parameter (not used in this implementation).
        monocrit (float, optional): Monotonicity criterion (not used in this implementation).

    Returns:
        ndarray: Array with cluster labels after combining leaves.
    """
    _ = hierarchy.is_valid_linkage(Z, throw=True)
    N = Z.shape[0] + 1
    T = hierarchy.fcluster(Z, t, criterion=criterion, depth=depth, R=R, monocrit=monocrit)
    L, M = hierarchy.leaders(Z, T)
    leaf_leaders = list(L[L < N])

    if len(leaf_leaders) == 0:
        return T

    max_cluster = T.max()

    for n, link in enumerate(
        Z[np.logical_or(*(np.in1d(Z[:, l], leaf_leaders) for l in range(2))), :2].astype("i")
    ):
        if n % 10 == 0:
            print(
                f"After {n} iterations, {len(leaf_leaders)} leaf leaders left with {len(np.unique(T))} total clusters"
            )

        if all([l in leaf_leaders for l in link]):
            max_cluster += 1
            T[link] = max_cluster
            _ = [leaf_leaders.remove(l) for l in link]
        elif any([l in leaf_leaders for l in link]):
            node_index = link[0] in leaf_leaders
            node, leaf = link[int(node_index)], link[int(~node_index)]

            if node in L:
                downstream_leaders = [node]
            else:
                tree = hierarchy.to_tree(Z, rd=True)[1][node]

                def check_node(node, nodes_to_check, downstream_leaders, L):
                    if node.id in L:
                        downstream_leaders.append(node.id)
                    else:
                        nodes_to_check.extend([node.left, node.right])
                    return nodes_to_check, downstream_leaders

                downstream_leaders = []
                nodes_to_check = [tree.left, tree.right]

                while len(nodes_to_check) > 0:
                    n_ = nodes_to_check.pop(0)
                    if all([s is None for s in [n_.left, n_.right]]):
                        raise ValueError(
                            "While traversing the tree, a leaf node was reached"
                            f", node {n_.id}. In theory this should not occur."
                        )
                    nodes_to_check, downstream_leaders = check_node(
                        n_, nodes_to_check, downstream_leaders, L
                    )

            max_cluster += 1
            merge_clusters = M[np.in1d(L, downstream_leaders)]
            T[np.in1d(T, merge_clusters)] = max_cluster
            T[leaf] = max_cluster
            _ = leaf_leaders.remove(leaf)
        else:
            continue

        L, M = hierarchy.leaders(Z, T)

        if len(leaf_leaders) == 0:
            break

    leaf_leaders = list(L[L < N])

    if len(leaf_leaders) == 0:
        print(
            f"All leaf leaders combined, resulting in {len(np.unique(T))} total clusters"
        )

        unique, inverse = np.unique(T, return_inverse=True)
        return np.arange(0, unique.shape[0])[inverse]
    else:
        raise ValueError(f"Failed to merge leaf leaders {leaf_leaders}")

def fcluster_multi_threshold(
    Z, ts, criterion="distance", depth=2, R=None, monocrit=None, min_cluster_size=2
):
    """
    Perform hierarchical clustering with multiple thresholds and combine small clusters.

    Parameters:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        ts (list of float): List of thresholds to apply.
        criterion (str): Criterion to use for cluster formation (default is "distance").
        depth (int): The depth of the tree to use for cluster formation.
        R (float, optional): Criterion parameter (not used in this implementation).
        monocrit (float, optional): Monotonicity criterion (not used in this implementation).
        min_cluster_size (int): Minimum size of clusters to keep.

    Returns:
        ndarray: Array with cluster labels after applying multiple thresholds and merging small clusters.
    """
    _ = hierarchy.is_valid_linkage(Z, throw=True)
    T_ = -1 * np.ones(Z.shape[0] + 1)
    N = len(ts)

    for n, t in enumerate(ts):
        T = hierarchy.fcluster(
            Z, t, criterion=criterion, depth=depth, R=R, monocrit=monocrit
        )
        uniques_, counts_ = np.unique(T, return_counts=True)
        if n == (N - 1):
            keep = T_ == -1
        else:
            keep = np.logical_and(
                np.in1d(T, uniques_[counts_ >= min_cluster_size]), T_ == -1
            )
        T_[keep] = T[keep] + T_.max()

    uniques_, inverse_ = np.unique(T_, return_inverse=True)
    return np.arange(0, uniques_.shape[0])[inverse_]

def merge_small_clusters(y, T, method="complete", min_cluster_size=2):
    """
    Internal function to merge small clusters based on distance matrix `y`.

    Parameters:
        y (ndarray): Distance matrix.
        T (ndarray): Cluster labels.
        method (str): Method for merging clusters ("complete" or "single").
        min_cluster_size (int): Minimum size of clusters to keep.

    Returns:
        ndarray: Array with updated cluster labels after merging small clusters.
    """
    T = T.copy()
    uniques, counts = np.unique(T, return_counts=True)
    size_sort = np.argsort(counts, kind="stable")[::-1]
    uniques, counts = uniques[size_sort], counts[size_sort]

    small_clusters = list(uniques[counts < min_cluster_size])

    if len(small_clusters) == 0:
        raise UserWarning(f"No clusters with fewer than {min_cluster_size} members found.")
        return T

    if method == "complete":
        distance_func = np.max
    elif method == "single":
        distance_func = np.min
    else:
        raise NotImplementedError(f"`method`={method} not implemented")

    while len(small_clusters) > 0:
        s_c = small_clusters.pop(0)
        merge = np.argmin(
            [
                distance_func(y[np.ix_(T == s_c, T == c)])
                for c in uniques[uniques != s_c]
            ]
        )
        T[T == s_c] = uniques[uniques != s_c][merge]

        uniques, counts = np.unique(T, return_counts=True)
        size_sort = np.argsort(counts, kind="stable")[::-1]
        uniques, counts = uniques[size_sort], counts[size_sort]

        small_clusters = list(uniques[counts < min_cluster_size])

    print(f"All small clusters combined, resulting in {len(uniques)} total clusters")
    uniques, inverse = np.unique(T, return_inverse=True)
    return np.arange(0, uniques.shape[0])[inverse]
