"""
01b_prepare_data_crispr_a.py
Normalize and prepare GSE152988 CRISPRa data (gene activation, not knockdown).
Outputs: data/crispr_a_processed.h5ad
"""
import scanpy as sc
import anndata as ad
import os

INPUT  = "GSE152988/TianKampmann2021_CRISPRa.h5ad"
OUTPUT = "data/crispr_a_processed.h5ad"
N_HVG  = 5000

os.makedirs("data", exist_ok=True)

print("Loading CRISPRa data...")
adata = ad.read_h5ad(INPUT)
print(f"  Raw shape: {adata.n_obs} cells x {adata.n_vars} genes")

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3",
                             layer="counts")
adata = adata[:, adata.var.highly_variable].copy()

adata.var["gene_name"] = adata.var_names.tolist()
adata.obs["condition"] = adata.obs["perturbation"].apply(
    lambda g: "ctrl" if g == "control" else f"{g}+ctrl"
)
adata.obs["cell_type"] = "iPSC-induced-neuron"

print(f"  After HVG: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"  Unique conditions: {adata.obs['condition'].nunique()}")
print(f"  Control cells: {(adata.obs['condition']=='ctrl').sum():,}")

adata.write_h5ad(OUTPUT)
print(f"Saved: {OUTPUT}")
