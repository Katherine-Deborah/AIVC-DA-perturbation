"""
01_prepare_data.py
Normalize and prepare GSE152988 CRISPRi data for GEARS and scGen.
Subsets to HVGs to fit in 4GB GPU memory.
Outputs: data/crispr_i_processed.h5ad
"""
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os

INPUT   = "GSE152988/TianKampmann2021_CRISPRi.h5ad"
OUTPUT  = "data/crispr_i_processed.h5ad"
N_HVG   = 5000   # number of highly variable genes

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading CRISPRi data...")
adata = ad.read_h5ad(INPUT)
print(f"  Raw shape: {adata.n_obs} cells x {adata.n_vars} genes")

# Store raw counts
adata.layers["counts"] = adata.X.copy()

# Normalize + log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("  Normalized and log1p-transformed")

# Select highly variable genes — reduces memory for GPU
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3",
                             layer="counts")
adata = adata[:, adata.var.highly_variable].copy()
print(f"  After HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")

# Add required columns
# GEARS needs: condition ('ctrl' or 'GENE+ctrl'), gene_name in var, cell_type in obs
adata.var["gene_name"] = adata.var_names.tolist()

adata.obs["condition"] = adata.obs["perturbation"].apply(
    lambda g: "ctrl" if g == "control" else f"{g}+ctrl"
)
adata.obs["cell_type"] = "iPSC-induced-neuron"

# Print summary
conds = adata.obs["condition"].value_counts()
print(f"\n  Total unique conditions: {len(conds)}")
print(f"  Control cells: {(adata.obs['condition'] == 'ctrl').sum():,}")
print(f"  Perturbed cells: {(adata.obs['condition'] != 'ctrl').sum():,}")
print(f"  Sample conditions:\n{conds.head(6).to_string()}")

adata.write_h5ad(OUTPUT)
print(f"\nSaved: {OUTPUT}")
