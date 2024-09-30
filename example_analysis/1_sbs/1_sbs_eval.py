import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from ops.qc import *
from ops.sbs_smk import Snake_sbs

# SET PARAMETERS
OUTPUT_FILES_DIR = "output"
DF_DESIGN_PATH = "input/pool10_design.csv"
READS_PATH = "output/10X_A1_Tile-50.reads.csv"
CELLS_PATH = "output/10X_A1_Tile-50.cells.csv"
SBS_INFO_PATH = "output/10X_A1_Tile-50.sbs_info.csv"


# Read barcodes
df_design = pd.read_csv(DF_DESIGN_PATH)
df_pool = df_design.query("dialout==[0,1]").drop_duplicates("sgRNA")
df_pool["prefix"] = df_pool.apply(lambda x: x.sgRNA[: x.prefix_length], axis=1)  # 13
barcodes = df_pool["prefix"]

# Load SBS output files
reads = pd.read_csv(READS_PATH)
cells = pd.read_csv(CELLS_PATH)
sbs_info = pd.read_csv(SBS_INFO_PATH)

# Generate plots
print("Generating plots...")
plot_mapping_vs_threshold(reads, barcodes, "peak")
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/mapping_vs_threshold_peak.png")
plt.close()

plot_mapping_vs_threshold(reads, barcodes, "Q_min")
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/mapping_vs_threshold_qmin.png")
plt.close()

plot_read_mapping_heatmap(reads, barcodes, shape="6W_sbs")
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/read_mapping_heatmap.png")
plt.close()

df_summary_one, _ = plot_cell_mapping_heatmap(
    cells,
    sbs_info,
    barcodes,
    mapping_to="one",
    mapping_strategy="gene_symbols",
    shape="6W_sbs",
    return_summary=True,
)
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/cell_mapping_heatmap_one.png")
plt.close()
df_summary_one.to_csv(f"{OUTPUT_FILES_DIR}/cell_mapping_heatmap_one.csv", index=False)

df_summary_any, _ = plot_cell_mapping_heatmap(
    cells,
    sbs_info,
    barcodes,
    mapping_to="any",
    mapping_strategy="gene_symbols",
    shape="6W_sbs",
    return_summary=True,
)
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/cell_mapping_heatmap_any.png")
plt.close()
df_summary_any.to_csv(f"{OUTPUT_FILES_DIR}/cell_mapping_heatmap_any.csv", index=False)

outliers = plot_reads_per_cell_histogram(cells, x_cutoff=20)
plt.savefig(f"{OUTPUT_FILES_DIR}/reads_per_cell_histogram.png")
plt.close()

outliers = plot_gene_symbol_histogram(cells, x_cutoff=30)
plt.gcf().savefig(f"{OUTPUT_FILES_DIR}/gene_symbol_histogram.png")
plt.close()

num_rows = len(sbs_info)
print(f"The number of cells extracted in the sbs step is: {num_rows}")

# Calculate and print mapped single gene statistics
print("Calculating mapped single gene statistics...")
cells["mapped_single_gene"] = cells.apply(
    lambda x: (
        True
        if (pd.notnull(x.gene_symbol_0) & pd.isnull(x.gene_symbol_1))
        | (x.gene_symbol_0 == x.gene_symbol_1)
        else False
    ),
    axis=1,
)
print(cells.mapped_single_gene.value_counts())

print("QC analysis completed.")
