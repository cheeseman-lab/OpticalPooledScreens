import os
import pandas as pd
import matplotlib.pyplot as plt
import ops.aggregate as agg 

# Set screen directories
ph_function_home = "/lab/barcheese01/screens"
ph_function_dataset = "aconcagua"
home = os.path.join(ph_function_home, ph_function_dataset)

# Create directories if they don't exist
qc_dir = os.path.join(home, 'aggregate_4', 'qc')
os.makedirs(qc_dir, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(qc_dir, filename))
    plt.close(fig)

population_feature = 'gene_symbol_0'
filter_single_gene = False
channels = ['dapi','tubulin','gh2ax','phalloidin']

print("Loading subsets of processed datasets...")
raw_df = agg.load_hdf_subset('merge_3/hdf/merged_final.hdf', n_rows=50000)
clean_df = agg.clean_cell_data(raw_df, population_feature, filter_single_gene=filter_single_gene)
transformed_df = agg.load_hdf_subset('aggregate_4/hdf/transformed_data.hdf', n_rows=clean_df.shape[0])
standardized_df = agg.load_hdf_subset('aggregate_4/hdf/standardized_data.hdf', n_rows=clean_df.shape[0])
dfs = {
    "clean": clean_df,
    "transformed": transformed_df, 
    "standardized": standardized_df
}

print("Creating feature distribution plots...")
cell_features = ['cell_{}_mean'.format(channel) for channel in channels]
nucleus_features = ['nucleus_{}_mean'.format(channel) for channel in channels]

plt.figure(figsize=(12, 6))
agg.plot_feature_distributions(dfs, cell_features, remove_clean=True)
save_plot(plt.gcf(), 'cell_feature_violins.png')

plt.figure(figsize=(12, 6))
agg.plot_feature_distributions(dfs, nucleus_features, remove_clean=True)
save_plot(plt.gcf(), 'nuclear_feature_violins.png')

print("Testing for missing values in final dataframes...")
mitotic = pd.read_csv('aggregate_4/csv/mitotic_gene_data.csv')
interphase = pd.read_csv('aggregate_4/csv/interphase_gene_data.csv')
all = pd.read_csv('aggregate_4/csv/all_gene_data.csv')

mitotic_missing = agg.test_missing_values(mitotic, 'mitotic')
interphase_missing = agg.test_missing_values(interphase, 'interphase')
all_missing = agg.test_missing_values(all, 'all')

mitotic_missing.to_csv(os.path.join(qc_dir, 'mitotic_missing.csv'), index=False)
interphase_missing.to_csv(os.path.join(qc_dir, 'interphase_missing.csv'), index=False)
all_missing.to_csv(os.path.join(qc_dir, 'all_missing.csv'), index=False)
