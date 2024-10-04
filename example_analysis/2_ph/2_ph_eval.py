import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ops.qc import *

# SET PARAMETERS
HDFS_DIR = "output/hdfs"
OUTPUT_FILES_DIR = "output/eval"

# create output directory if it doesn't exist
os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)

# helper function to load and concatenate hdfs
def load_and_concatenate_hdfs(hdf_well_load_path):
    # Find all HDF files matching the provided path pattern
    hdf_files = glob.glob(hdf_well_load_path)
    
    # Load each HDF file into a pandas DataFrame
    dfs = [pd.read_hdf(file) for file in hdf_files]
    
    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    return concatenated_df

# Concatenate files
print("Concatenating files...")
phenotype_info = load_and_concatenate_hdfs(f"{HDFS_DIR}/phenotype_info_*.hdf")
min_cp_phenotype = load_and_concatenate_hdfs(f"{HDFS_DIR}/min_cp_phenotype_*.hdf")

# Generate plots
print("Generating plots...")

df_summary, _ = plot_count_heatmap(phenotype_info, shape='6W_sbs', return_summary=True)
df_summary.to_csv(f"{OUTPUT_FILES_DIR}/phenotype_count_heatmap.csv", index=False)
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/phenotype_count_heatmap.png")
plt.close()

# Plot feature heatmaps for each cellular marker
features = ['cell_dapi_min', 'cell_cenpa_min', 'cell_coxiv_min', 'cell_wga_min']
for feature in features:
    df_summary, _ = plot_feature_heatmap(min_cp_phenotype, feature=feature, shape='6W_sbs', return_summary=True)
    df_summary.to_csv(f"{OUTPUT_FILES_DIR}/{feature}_heatmap.csv", index=False)
    plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/{feature}_heatmap.png")
    plt.close()

num_rows = len(phenotype_info)
print(f"The number of cells extracted in the phenotype step is: {num_rows}")

print("QC analysis completed.")

print("Concatenating large cp files...")
cp_phenotype = load_and_concatenate_hdfs(f"{HDFS_DIR}/cp_phenotype_*.hdf")
cp_phenotype.to_hdf(os.path.join(HDFS_DIR, 'output/hdfs/cp_phenotype.hdf'), key='cp_phenotype', mode='w', format='table')
