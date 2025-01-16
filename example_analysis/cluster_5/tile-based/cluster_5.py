import os
import pandas as pd
import openpyxl
import json
import ops.cluster as cluster
import ops.plotting as plotting

print("Setting up configuration parameters...")

cluster_function_home = "/lab/barcheese01/screens"
cluster_function_dataset = "denali"
home = os.path.join(cluster_function_home, cluster_function_dataset)

os.makedirs("cluster_5/csv", exist_ok=True)
os.makedirs("cluster_5/plots", exist_ok=True)
output_dir = "cluster_5"

control_prefix = 'nontargeting'
cutoffs = {
    'mitotic': 0,
    'interphase': 3,
    'all': 3
}
channels = ['dapi', 'coxiv', 'cenpa', 'wga']
channel_pairs = ['all', ('dapi', 'coxiv'), ('dapi', 'cenpa'), ('dapi', 'wga')]

# Analysis parameters
correlation_threshold = 0.99
variance_threshold = 0.001
min_unique_values = 5
leiden_resolutions = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]

# Automatically generate feature pairs
feature_pairs = []

# Generate channel-specific comparisons
for i, channel1 in enumerate(channels):
    # Compare each channel's nuclear measurement with DAPI
    if channel1 != 'dapi':
        feature_pairs.extend([
            (f'nucleus_dapi_mean', f'nucleus_{channel1}_mean'),
            (f'cell_dapi_mean', f'cell_{channel1}_mean')
        ])
    
    # Compare cellular vs nuclear measurements for each channel
    feature_pairs.append((f'cell_{channel1}_mean', f'nucleus_{channel1}_mean'))

# Add morphology comparisons
morphology_pairs = [
    ('cell_area', 'nucleus_area'),
]
feature_pairs.extend(morphology_pairs)

print("Generated feature pairs:", feature_pairs)
    
# Load datasets
print("Loading datasets...")
mitotic_path = "aggregate_4/csv/mitotic_gene_data.csv"
interphase_path = "aggregate_4/csv/interphase_gene_data.csv"
all_path = "aggregate_4/csv/all_gene_data.csv"

df_mitotic, df_interphase, df_all = cluster.load_gene_level_data(mitotic_path, interphase_path, all_path)
datasets = {
    'mitotic': df_mitotic,
    'interphase': df_interphase,
    'all': df_all
}

# Calculate mitotic percentage
print("Calculating mitotic percentage...")
mitotic_stats = cluster.calculate_mitotic_percentage(datasets['mitotic'], datasets['interphase'])
mitotic_stats.to_csv(f"{output_dir}/csv/mitotic_stats.csv", index=False)

# Modify your main code to include channel-pair-specific directories
for channel_pair in channel_pairs:
    # Create channel-specific output directory
    if channel_pair == 'all':
        pair_dir = os.path.join(output_dir, "all_channels")
    else:
        pair_dir = os.path.join(output_dir, f"channels_{'_'.join(channel_pair)}")
    
    os.makedirs(os.path.join(pair_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(pair_dir, "plots"), exist_ok=True)
    
    # Filter datasets for current channel combination
    current_datasets = {
        'mitotic': cluster.split_channels(datasets['mitotic'], channel_pair, channels),
        'interphase': cluster.split_channels(datasets['interphase'], channel_pair, channels),
        'all': cluster.split_channels(datasets['all'], channel_pair, channels),
    }
    
    # Process each dataset with the current channel selection
    for dataset_type, df in current_datasets.items():
        print(f"\nProcessing {dataset_type} dataset with channels {channel_pair}...")
        
        # Generate feature pairs specific to current channel selection
        current_feature_pairs = []
        
        if channel_pair != 'all':
            # Generate channel-specific comparisons for the selected channels
            for channel in channel_pair:
                if channel != 'dapi' and 'dapi' in channel_pair:
                    current_feature_pairs.extend([
                        (f'nucleus_dapi_mean', f'nucleus_{channel}_mean'),
                        (f'cell_dapi_mean', f'cell_{channel}_mean')
                    ])
                current_feature_pairs.append((f'cell_{channel}_mean', f'nucleus_{channel}_mean'))
        else:
            current_feature_pairs = feature_pairs.copy()
        
        # Add morphology comparisons
        current_feature_pairs.extend(morphology_pairs)
        
        # Plot cell histogram with updated path
        print(f"Plotting {dataset_type} cell histogram...")
        plotting.plot_cell_histogram(
            df, 
            cutoff=cutoffs[dataset_type], 
            bins=300, 
            save_plot_path=f"{pair_dir}/plots/{dataset_type}_cell_count_histogram.pdf"
        )
        
        # Remove low number genes and missing features
        print(f"Cleaning {dataset_type} dataset...")
        df = cluster.remove_low_number_genes(df, min_cells=cutoffs[dataset_type])
        df = cluster.remove_missing_features(df)
        df.to_csv(f"{pair_dir}/csv/{dataset_type}_clean.csv", index=False)

        # Generate a ranked dataframe of features
        print(f"Generating ranked {dataset_type} dataset...")
        df_ranked = cluster.rank_transform(df)
        df_ranked.to_csv(f"{pair_dir}/csv/{dataset_type}_ranked.csv", index=False)

        # Generate feature plots only if channel_pair is 'all'
        if channel_pair == 'all':
            print(f"Generating {dataset_type} feature plots...")
            for x_feature, y_feature in feature_pairs:
                plotting.two_feature(
                    df,
                    x=x_feature, 
                    y=y_feature,
                    control_query='gene_symbol_0.str.startswith("nontargeting")',
                    # annotate_query=f'(abs({x_feature} - {y_feature}) > 0.9) & ~gene_symbol_0.str.startswith("nontargeting")',
                    # annotate_labels='gene_symbol_0',
                    control_kwargs={'alpha': 0.5, 'color': 'gray'},  
                    annotate_kwargs={'alpha': 1.0},  
                    save_plot_path=f"{pair_dir}/plots/{dataset_type}_{x_feature}_{y_feature}.pdf"
                )
        
        # Feature selection
        filtered_file_path = f"{pair_dir}/csv/{dataset_type}_filtered.csv"
        removed_features_path = f"{pair_dir}/csv/{dataset_type}_removed_features.txt"
        
        if os.path.exists(filtered_file_path):
            print(f"Loading existing filtered data for {dataset_type}...")
            df_filtered = pd.read_csv(filtered_file_path)
            with open(removed_features_path, 'r') as f:
                removed_features = json.load(f)
        else:
            print(f"Performing feature selection for {dataset_type}...")
            df_filtered, removed_features = cluster.select_features(
                df, 
                correlation_threshold=correlation_threshold, 
                variance_threshold=variance_threshold, 
                min_unique_values=min_unique_values
            )
            df_filtered.to_csv(filtered_file_path, index=False)
            with open(removed_features_path, "w") as f:
                f.write(json.dumps(removed_features))
        
        # Normalization
        print(f"Performing normalization for {dataset_type}...")
        df_norm = cluster.normalize_to_controls(df_filtered, control_prefix)
        
        # Hierarchical clustering and heatmap
        print(f"Generating heatmap for {dataset_type}...")
        plotting.heatmap(
            df_norm, 
            vmin=-5, 
            vmax=5,
            cmap='RdBu_r',
            label_fontsize=5,
            rasterized=True,
            row_cluster=True,  
            col_cluster=True,  
            dendrogram_ratio=0.05,  
            colors_ratio=0.03,
            alternate_ticks=(True, True),
            xticklabel_kwargs={'ha': 'right', 'va': 'top'},
            save_plot_path=f"{pair_dir}/plots/{dataset_type}_heatmap.pdf"
        )
        
        # PCA
        print(f"Performing PCA for {dataset_type}...")
        df_pca, n_components, pca = cluster.perform_pca_analysis(
            df_norm, 
            save_plot_path=f"{pair_dir}/plots/{dataset_type}_pca_variance_plot.png"
        )
        
        # Plot PCA heatmap
        plotting.heatmap(
            df_pca, 
            vmin=-5, 
            vmax=5,
            cmap='RdBu_r',
            label_fontsize=5,
            rasterized=True,
            row_cluster=True,  
            col_cluster=True,  
            dendrogram_ratio=0.05,  
            colors_ratio=0.03,
            alternate_ticks=(True, True),
            xticklabel_kwargs={'ha': 'right', 'va': 'top'},
            save_plot_path=f"{pair_dir}/plots/{dataset_type}_pca_heatmap.pdf"
        )
        
        # PHATE and subsequent analysis with different resolutions
        print(f"Performing PHATE and resolution analysis for {dataset_type}...")
        for resolution in leiden_resolutions:
            resolution_dir = os.path.join(pair_dir, f"resolution_{resolution}")
            os.makedirs(os.path.join(resolution_dir, "csv"), exist_ok=True)
            os.makedirs(os.path.join(resolution_dir, "plots"), exist_ok=True)
            
            # PHATE and Leiden clustering with current resolution
            print(f"Performing PHATE and Leiden clustering (resolution={resolution}) for {dataset_type}...")
            df_phate = cluster.phate_leiden_pipeline(df_pca, resolution=resolution)
            df_phate.to_csv(f"{resolution_dir}/csv/{dataset_type}_clustering.csv")
            
            # Final PHATE plot
            print(f"Generating PHATE plot for {dataset_type}, resolution {resolution}...")
            plotting.dimensionality_reduction(
                df_phate,
                x='PHATE_0',
                y='PHATE_1',
                control_query='gene_symbol_0.str.startswith("nontargeting")',
                control_color='lightgray',
                control_legend=True,
                label_query='~gene_symbol_0.str.startswith("nontargeting")',
                label_hue='cluster',
                label_palette='husl',
                s=25, 
                hide_axes=False,
                label_legend=False,
                legend_kwargs={'loc': 'center left', 'bbox_to_anchor': (1, 0.5)},
                save_plot_path=f'{resolution_dir}/plots/{dataset_type}_phate_plot.pdf'
            )

            # Make the databases dir in the output directory
            databases_dir = f"{output_dir}/databases"
            os.makedirs(databases_dir, exist_ok=True)

            # Merge PHATE results with UniProt data
            print(f"Merging PHATE results with UniProt data for {dataset_type}, resolution {resolution}...")
            df_merged = cluster.merge_phate_uniprot(df_phate, databases_dir)
            df_merged.to_csv(f"{resolution_dir}/csv/{dataset_type}_clustering_uniprot.csv")

            # Create cluster gene table
            print(f"Creating cluster gene table for {dataset_type}, resolution {resolution}...")
            cluster_gene_table = cluster.create_cluster_gene_table(
                df_merged, 
                columns_to_combine=['gene_symbol_0', 'STRING']
            )
            
            # Analyze differential features
            print(f"Analyzing differential features for {dataset_type}, resolution {resolution}...")
            cluster_gene_table, diff_results = cluster.analyze_differential_features(
                cluster_gene_table, 
                df, 
                n_top=5
            )
            
            # Process interactions and get enrichment results
            print(f"Processing interactions for {dataset_type}, resolution {resolution}...")
            cluster_gene_table, global_metrics = cluster.process_interactions(cluster_gene_table, databases_dir)
            
            # Save results with dataset-specific prefixes in csv directory
            cluster_gene_table.to_csv(f"{resolution_dir}/csv/{dataset_type}_cluster_gene_table.csv", index=False)
            
            # Save global metrics
            with open(f"{resolution_dir}/csv/{dataset_type}_global_metrics.txt", "w") as f:
                for key, value in global_metrics.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Enrichment analysis complete for {dataset_type}, {channel_pair}, resolution {resolution}. Results saved to {resolution_dir}/csv/")

        print(f"Merging clustering results across resolutions for {dataset_type}, {channel_pair}...")
        merged_clusters = cluster.merge_resolution_clusters(
            output_dir=output_dir,
            dataset_type=dataset_type,
            channel_pair=channel_pair,
            resolutions=leiden_resolutions
        )
        merged_clusters.to_csv(f"{pair_dir}/csv/{dataset_type}_merged_clustering.csv", index=False)

df_metrics = cluster.aggregate_resolution_metrics(
    output_dir = output_dir,
    dataset_types=['mitotic', 'interphase', 'all'],
    channel_pairs=channel_pairs,
    leiden_resolutions=leiden_resolutions
)

df_metrics.to_csv(f"{output_dir}/csv/clustering_summary.csv")

print("\nAnalysis complete! All results have been saved to the cluster_5 directory.")