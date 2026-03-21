# CLAUDE.md — scFoundation + GEARS Pipeline for DA Neuron Perturbation Prediction

## Project Overview

This pipeline runs **scFoundation + GEARS** for perturbation prediction on two single-cell datasets from a dopaminergic (DA) neuron virtual cell modeling project. The goal is to predict how gene perturbations shift cells toward a DA neuron transcriptional state, enabling systematic gene ranking for experimental validation in Parkinson's disease research.

**Architecture:** scFoundation acts as a frozen feature extractor generating gene context embeddings (shape: `cells × 19264 × 512`). These embeddings are fed as node features into GEARS (a graph neural network), which predicts post-perturbation gene expression.

---

## Datasets

### GSE152988 — Primary Training Dataset
- **Cell type:** iPSC-derived neurons / DA neuron-like cells (e.g., LUHMES line)
- **Species:** Human
- **Files:** 6 `.h5` files (10x Genomics format)
  - CRISPRi: `GSM4632022`, `GSM4632023`, `GSM4632025`, `GSM4632026` (4 files, ~58,657 cells)
  - CRISPRa: `GSM4632027`, `GSM4632028` (2 files, ~37,980 cells)
- **Total cells:** ~96,637 cells × 33,538 genes
- **Perturbations:** 184 genes for CRISPRi, 100 genes for CRISPRa
- **DA markers:** 67/71 DA markers present; ~12,997 cells (13.4%) score highly for DA markers
- **Key canonical DA genes found:** TH, DDC, SLC6A3 (DAT), SLC18A2 (VMAT2), NR4A2 (NURR1), DRD1-5
- **Note:** No explicit per-cell perturbation labels in metadata. Perturbation type (CRISPRi vs CRISPRa) must be extracted from filenames.

### GSE124703 — Secondary Training Dataset
- **Cell type:** iPSC and iPSC-derived neurons (Day 7+)
- **Species:** Human
- **Files:** 4 MTX triplets + 4 sgRNA mapping files
  - CROP-seq samples: iPSC_1, iPSC_2 (GSM3543618-619), neuron_1, neuron_2 (GSM3543622-623)
  - sgRNA mapping: GSM3543620-621 (iPSC), GSM3543624-625 (neuron)
  - Bulk Quant-seq (Day 14/21/28): GSM3543612-617 — **SKIP these for cell count / perturbation tasks**
- **Total CROP-seq cells:** ~real filtered count TBD (737,280 is unfiltered; real count ~10k–50k per sample)
- **Perturbations:** 57 guide sequences → ~20–25 gene targets (CRISPRi knockdowns)
- **Cell contexts:** iPSC stage AND Day 7 neuron stage — same guides in both
- **DA markers:** 70/71 DA markers present in gene panel; but TH=10.99 CPM, SLC6A3=0 — these are generic neurons, NOT confirmed DA neurons
- **sgRNA → gene mapping:** NOT in GEO deposit. Refer to Supplementary Table S4 of Tian et al. 2019 (Neuron, PMID: 31422865) for the 57 guide → gene name mapping.

### GSE140231 — DA Reference Dataset (for DA identity labels)
- **Use:** Define DA neuron identity manifold; prepare DA neuron labels for training
- **Cell type:** DA neurons + SN cell types from human substantia nigra
- **Species:** Human + Mouse
- **Cells:** ~20k total (~1k per individual)
- **DA markers:** 70/71 found
- **Note:** This is the TA-recommended reference for DA neuron labels. Preprocess first to get clean DA labels before training.

---

## DA Marker Gene List (71 genes)

Located in `DA_gene_dataset.csv` (column: `gene_symbol`). Key canonical markers used for scoring:

| Category | Genes |
|----------|-------|
| Biosynthesis | TH, DDC, GCH1, PTS, SPR, QDPR |
| Transporters | SLC6A3 (DAT), SLC18A2 (VMAT2), SLC18A1 |
| Catabolism | MAOA, MAOB, COMT, ALDH2, ALDH1A1 |
| TFs (DA identity) | NR4A2 (NURR1), LMX1A, LMX1B, FOXA2, PITX3, EN1, EN2, ASCL1 |
| Receptors | DRD1, DRD2, DRD3, DRD4, DRD5 |
| PD genes | SNCA, LRRK2, GBA1, PARK2, PINK1, PARK7, UCHL1 |
| Mitochondria | NDUFS1, NDUFV1, COX4I1, ATP5A1, SOD2 |

**Missing from datasets (zero-pad when aligning):** GBA1, PARK2, ATP5A1, GPX1

---

## Data Split Strategy

**Golden rule: Split by biological unit (condition/guide), NOT by individual cell.**

### GSE152988
```
Train:  CRISPRi lib1 + lib2 + CRISPRa lib1  (~75%)
Val:    CRISPRi lib3                          (~10%)
Test:   CRISPRi lib4 + CRISPRa lib2          (~15%)
```

### GSE124703
```
# Option A — Split by cell type context:
Train:  iPSC_1 + neuron_1
Val:    iPSC_2
Test:   neuron_2

# Option B — Split by guide identity (stricter, preferred for perturbation eval):
Train:  80% of 57 guides (~45 guides), both cell types
Val:    10% of guides (~6 guides)
Test:   10% of guides (~6 guides) — completely unseen perturbations
```

---

## Evaluation Metrics

### Cell Type Annotation (DA vs non-DA)
- Macro F1, AUROC (handles class imbalance — DA cells are ~13% of GSE152988)
- Average Silhouette Width (ASW) in embedding space
- Adjusted Rand Index (ARI) vs DA marker-derived labels
- Biological sanity check: TH / SLC6A3 / NR4A2 expression rank within predicted DA clusters

### Perturbation Prediction
- **Pearson correlation** (overall, all genes)
- **Pearson correlation top-20 DEGs** — standard GEARS metric
- MSE / RMSE
- R² (variance explained)
- MMD (Maximum Mean Discrepancy) — distributional similarity

Report separately for:
- Seen perturbations (guides in training) → memorization
- Unseen perturbations (held-out guides) → true generalization

---

## Prerequisites

### Hardware
- **Minimum:** 1 GPU with 40GB VRAM (A100 preferred for frozen mode)
- **Full fine-tune:** 80GB+ VRAM
- **Alternatives:** Google Colab Pro (A100), university HPC cluster

### Software
- Python 3.8+
- CUDA 11.7+ (match your GPU)
- conda (recommended for environment management)

---

## Step-by-Step Instructions

### Step 1 — Check hardware and clone repo

```bash
# Verify GPU
nvidia-smi

# Clone scFoundation
git clone https://github.com/biomap-research/scFoundation.git
cd scFoundation
```

### Step 2 — Set up conda environment

```bash
conda create -n scfoundation python=3.8 -y
conda activate scfoundation

# Install PyTorch (adjust cuda version to match your system)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyG dependencies
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# Install GEARS and single-cell libraries
pip install cell-gears
pip install scanpy anndata pandas numpy scipy scikit-learn
pip install huggingface_hub

# Install any remaining requirements from repo
pip install -r requirements.txt 2>/dev/null || true
```

### Step 3 — Download pretrained scFoundation weights

```bash
mkdir -p model/models
cd model/models

python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="biomap-research/scFoundation",
    filename="models.ckpt",
    local_dir="."
)
print("Model weights downloaded successfully.")
EOF

cd ../../  # back to scFoundation root
```

### Step 4 — Prepare data directories

```bash
mkdir -p data results/GSE152988 results/GSE124703
# Place your downloaded files:
# data/GSE152988/  → the 6 .h5 files
# data/GSE124703/  → the MTX triplets + sgRNA mapping files
# data/DA_gene_dataset.csv → DA marker gene list
```

### Step 5 — Preprocess GSE152988

Save as `prepare_GSE152988.py` and run:

```python
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import glob, os

H5_FOLDER = "./data/GSE152988/"
OUTPUT    = "./data/GSE152988_processed.h5ad"

files = sorted(glob.glob(os.path.join(H5_FOLDER, "*.h5")))
print(f"Found {len(files)} .h5 files")

adatas = []
for f in files:
    adata = sc.read_10x_h5(f)
    adata.var_names_make_unique()
    fname = os.path.basename(f)

    # Extract perturbation type from filename
    if "CRISPRi" in fname:
        adata.obs["perturbation_type"] = "CRISPRi"
    elif "CRISPRa" in fname:
        adata.obs["perturbation_type"] = "CRISPRa"

    adata.obs["source_file"] = fname
    # Encode condition as CRISPRi_lib1, CRISPRa_lib1 etc for GEARS
    lib_num = fname.split("lib")[-1].replace(".h5","")
    adata.obs["condition"] = f"{adata.obs['perturbation_type'][0]}_lib{lib_num}"
    adatas.append(adata)
    print(f"  {fname}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

adata_all = ad.concat(adatas, label="batch")
adata_all.var_names_make_unique()

# QC filtering
sc.pp.filter_cells(adata_all, min_genes=200)
sc.pp.filter_genes(adata_all, min_cells=3)
print(f"\nAfter QC: {adata_all.n_obs:,} cells x {adata_all.n_vars:,} genes")

# Store raw counts BEFORE normalization (required by scFoundation)
adata_all.layers["counts"] = adata_all.X.copy()

# Normalize and log-transform
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)

# DA scoring using marker genes
da_markers_df = pd.read_csv("./data/DA_gene_dataset.csv")
da_genes = da_markers_df["gene_symbol"].dropna().unique().tolist()
overlap = [g for g in da_genes if g in set(adata_all.var_names)]
print(f"\nDA markers found: {len(overlap)}/{len(da_genes)}")
sc.tl.score_genes(adata_all, gene_list=overlap, score_name="DA_score")
mean_s = adata_all.obs["DA_score"].mean()
std_s  = adata_all.obs["DA_score"].std()
adata_all.obs["DA_label"] = (adata_all.obs["DA_score"] > mean_s + std_s).astype(int)
n_da = adata_all.obs["DA_label"].sum()
print(f"DA marker-high cells: {n_da:,} ({100*n_da/adata_all.n_obs:.1f}%)")

adata_all.write_h5ad(OUTPUT)
print(f"\nSaved: {OUTPUT}")
print(f"Conditions: {adata_all.obs['condition'].value_counts().to_dict()}")
```

```bash
python prepare_GSE152988.py
```

### Step 6 — Preprocess GSE124703

Save as `prepare_GSE124703.py` and run:

```python
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os

DATA_FOLDER = "./data/GSE124703/"
OUTPUT      = "./data/GSE124703_processed.h5ad"

# CROP-seq sample triplets (matrix, barcodes, genes)
samples = {
    "iPSC_1":   ("GSM3543618_iPSC_1_matrix.mtx.gz",
                 "GSM3543618_iPSC_1_barcodes.tsv.gz",
                 "GSM3543618_iPSC_1_genes.tsv.gz"),
    "iPSC_2":   ("GSM3543619_iPSC_2_matrix.mtx.gz",
                 "GSM3543619_iPSC_2_barcodes.tsv.gz",
                 "GSM3543619_iPSC_2_genes.tsv.gz"),
    "neuron_1": ("GSM3543622_neuron_1_matrix.mtx.gz",
                 "GSM3543622_neuron_1_barcodes.tsv.gz",
                 "GSM3543622_neuron_1_genes.tsv.gz"),
    "neuron_2": ("GSM3543623_neuron_2_matrix.mtx.gz",
                 "GSM3543623_neuron_2_barcodes.tsv.gz",
                 "GSM3543623_neuron_2_genes.tsv.gz"),
}

sgrna_files = {
    "iPSC_1":   "GSM3543620_iPSC_1_sgRNA_mapping.txt.gz",
    "iPSC_2":   "GSM3543621_iPSC_2_sgRNA_mapping.txt.gz",
    "neuron_1": "GSM3543624_neuron_1_sgRNA_mapping.txt.gz",
    "neuron_2": "GSM3543625_neuron_2_sgRNA_mapping.txt.gz",
}

# IMPORTANT: Load sgRNA barcode → gene name mapping from:
# Tian et al. 2019 Neuron Supplementary Table S4
# Format: {barcode_sequence: gene_name}
# Replace with actual mapping once you have the supplement
GUIDE_TO_GENE = {}  # e.g. {"GGCTCCAGTTAACGCAGTCG": "APP", ...}
# If mapping unavailable, use guide barcode directly as condition label

adatas = []
for sample, (mat_f, bc_f, gene_f) in samples.items():
    print(f"\nLoading {sample}...")
    adata = sc.read_mtx(os.path.join(DATA_FOLDER, mat_f)).T
    barcodes = pd.read_csv(
        os.path.join(DATA_FOLDER, bc_f), header=None)[0].values
    genes = pd.read_csv(
        os.path.join(DATA_FOLDER, gene_f), header=None, sep="\t")

    adata.obs_names = barcodes
    adata.var["gene_id"]     = genes[0].values
    adata.var["gene_symbol"] = genes[1].values if genes.shape[1] > 1 else genes[0].values
    adata.var_names          = adata.var["gene_symbol"].values
    adata.var_names_make_unique()

    adata.obs["sample"]    = sample
    adata.obs["cell_type"] = "iPSC" if "iPSC" in sample else "neuron_Day7"

    # Load sgRNA mapping — assign condition per cell
    sgrna_df = pd.read_csv(
        os.path.join(DATA_FOLDER, sgrna_files[sample]), sep="\t")
    sgrna_df = sgrna_df[sgrna_df["cell"].isin(set(barcodes))]
    sgrna_df = sgrna_df[sgrna_df["umi_count"] > 0]
    sgrna_df = sgrna_df[~sgrna_df["barcode"].str.startswith("unprocessed_")]

    # Top guide per cell (highest UMI)
    top_guide = (sgrna_df.sort_values("umi_count", ascending=False)
                         .groupby("cell")["barcode"].first())

    adata.obs["guide_barcode"] = adata.obs_names.map(top_guide)
    # Map to gene name if mapping available, else use barcode
    adata.obs["condition"] = (
        adata.obs["guide_barcode"]
             .map(GUIDE_TO_GENE)
             .fillna(adata.obs["guide_barcode"])
             .fillna("ctrl")
    )

    adatas.append(adata)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"  Cells with guide: {adata.obs['condition'].ne('ctrl').sum():,}")

adata_all = ad.concat(adatas, label="batch")
adata_all.var_names_make_unique()

# QC
sc.pp.filter_cells(adata_all, min_genes=200)
sc.pp.filter_genes(adata_all, min_cells=3)
print(f"\nAfter QC: {adata_all.n_obs:,} cells x {adata_all.n_vars:,} genes")

# Store raw counts
adata_all.layers["counts"] = adata_all.X.copy()
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)

adata_all.write_h5ad(OUTPUT)
print(f"Saved: {OUTPUT}")
print(f"Unique conditions: {adata_all.obs['condition'].nunique()}")
print(f"Cell type breakdown:\n{adata_all.obs['cell_type'].value_counts()}")
```

```bash
python prepare_GSE124703.py
```

### Step 7 — Align genes to scFoundation's 19,264-gene vocabulary

scFoundation was pretrained on exactly **19,264 human genes**. Your data must match this exactly.

```python
# save as align_genes.py
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Gene list file is inside the cloned repo at:
GENE_LIST = "./model/OS_scRNA_gene_index.19264.tsv"

scf_genes = pd.read_csv(GENE_LIST, sep="\t", header=None)[0].tolist()
print(f"scFoundation vocab: {len(scf_genes)} genes")

def align_to_scfoundation(adata, scf_genes, output_path):
    common  = [g for g in scf_genes if g in set(adata.var_names)]
    missing = [g for g in scf_genes if g not in set(adata.var_names)]
    print(f"  Genes in common:         {len(common)}/{len(scf_genes)}")
    print(f"  Missing (zero-padded):   {len(missing)}")

    adata_common = adata[:, common].copy()

    # Zero-pad missing genes
    n_cells   = adata_common.n_obs
    zero_block = sp.csr_matrix((n_cells, len(missing)))
    missing_adata = ad.AnnData(
        X=zero_block,
        obs=adata_common.obs.copy(),
        var=pd.DataFrame(index=missing)
    )
    adata_full = ad.concat([adata_common, missing_adata], axis=1)
    adata_full = adata_full[:, scf_genes].copy()  # reorder to scFoundation order

    assert adata_full.n_vars == 19264, \
        f"Expected 19264, got {adata_full.n_vars}"
    print(f"  Output shape: {adata_full.n_obs} x {adata_full.n_vars}")

    adata_full.write_h5ad(output_path)
    return adata_full

for dataset, in_path, out_path in [
    ("GSE152988", "./data/GSE152988_processed.h5ad", "./data/GSE152988_aligned.h5ad"),
    ("GSE124703", "./data/GSE124703_processed.h5ad", "./data/GSE124703_aligned.h5ad"),
]:
    print(f"\nAligning {dataset}...")
    adata = sc.read_h5ad(in_path)
    align_to_scfoundation(adata, scf_genes, out_path)

print("\nGene alignment complete.")
```

```bash
python align_genes.py
```

### Step 8 — Train/val/test split

```python
# save as split_data.py
import scanpy as sc
import numpy as np

def split_by_condition(adata, dataset_name,
                       train_frac=0.70, val_frac=0.15, seed=42):
    np.random.seed(seed)
    conditions = adata.obs["condition"].unique()
    non_ctrl   = [c for c in conditions if c != "ctrl"]
    np.random.shuffle(non_ctrl)

    n       = len(non_ctrl)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_conds = set(non_ctrl[:n_train]) | {"ctrl"}
    val_conds   = set(non_ctrl[n_train:n_train+n_val])
    test_conds  = set(non_ctrl[n_train+n_val:])

    train = adata[adata.obs["condition"].isin(train_conds)].copy()
    val   = adata[adata.obs["condition"].isin(val_conds)].copy()
    test  = adata[adata.obs["condition"].isin(test_conds)].copy()

    print(f"\n=== {dataset_name} splits ===")
    print(f"  Train: {train.n_obs:,} cells | {len(train_conds)} conditions")
    print(f"  Val:   {val.n_obs:,} cells  | {len(val_conds)} conditions")
    print(f"  Test:  {test.n_obs:,} cells | {len(test_conds)} conditions")

    train.write_h5ad(f"./data/{dataset_name}_train.h5ad")
    val.write_h5ad(f"./data/{dataset_name}_val.h5ad")
    test.write_h5ad(f"./data/{dataset_name}_test.h5ad")

for dataset in ["GSE152988", "GSE124703"]:
    adata = sc.read_h5ad(f"./data/{dataset}_aligned.h5ad")
    split_by_condition(adata, dataset)

print("\nSplits saved.")
```

```bash
python split_data.py
```

### Step 9 — Run scFoundation + GEARS training

```bash
cd scFoundation/GEARS

# Create training script for GSE152988
cat > run_sh/run_GSE152988.sh << 'EOF'
#!/bin/bash
mkdir -p ../../results/GSE152988

python main.py \
    --data_path         ../../data/GSE152988_train.h5ad \
    --val_data_path     ../../data/GSE152988_val.h5ad \
    --test_data_path    ../../data/GSE152988_test.h5ad \
    --model_type        maeautobin \
    --bin_set           autobin_resolution_append \
    --finetune_method   frozen \
    --singlecell_model_path ../../model/models/models.ckpt \
    --batch_size        16 \
    --epochs            15 \
    --lr                1e-4 \
    --seed              42 \
    --result_dir        ../../results/GSE152988/ \
    2>&1 | tee ../../results/GSE152988/train.log

echo "GSE152988 training complete"
EOF

chmod +x run_sh/run_GSE152988.sh
bash run_sh/run_GSE152988.sh
```

For GSE124703:
```bash
cat > run_sh/run_GSE124703.sh << 'EOF'
#!/bin/bash
mkdir -p ../../results/GSE124703

python main.py \
    --data_path         ../../data/GSE124703_train.h5ad \
    --val_data_path     ../../data/GSE124703_val.h5ad \
    --test_data_path    ../../data/GSE124703_test.h5ad \
    --model_type        maeautobin \
    --bin_set           autobin_resolution_append \
    --finetune_method   frozen \
    --singlecell_model_path ../../model/models/models.ckpt \
    --batch_size        16 \
    --epochs            15 \
    --lr                1e-4 \
    --seed              42 \
    --result_dir        ../../results/GSE124703/ \
    2>&1 | tee ../../results/GSE124703/train.log

echo "GSE124703 training complete"
EOF

chmod +x run_sh/run_GSE124703.sh
bash run_sh/run_GSE124703.sh
```

### Step 10 — Evaluate and collect results

```python
# save as evaluate.py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import json, os

def compute_metrics(pred, true, top_k=20):
    metrics = {}

    # Overall Pearson
    r_all, _ = pearsonr(pred.flatten(), true.flatten())
    metrics["pearson_all"] = round(float(r_all), 4)

    # MSE
    metrics["mse"] = round(float(mean_squared_error(true, pred)), 4)

    # Top-20 DEG Pearson (standard GEARS metric)
    gene_var  = true.var(axis=0)
    top_idx   = np.argsort(gene_var)[-top_k:]
    r_top, _  = pearsonr(pred[:, top_idx].flatten(), true[:, top_idx].flatten())
    metrics[f"pearson_top{top_k}DEG"] = round(float(r_top), 4)

    # R²
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    metrics["r2"] = round(float(1 - ss_res / ss_tot), 4)

    return metrics

for dataset in ["GSE152988", "GSE124703"]:
    results_dir = f"./results/{dataset}/"
    print(f"\n=== {dataset} ===")
    try:
        pred = np.load(os.path.join(results_dir, "predictions.npy"))
        true = np.load(os.path.join(results_dir, "ground_truth.npy"))
        metrics = compute_metrics(pred, true)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {results_dir}metrics.json")
    except FileNotFoundError as e:
        print(f"  Results not found: {e}")
        print("  Training may still be running — check train.log")
```

```bash
cd ../../  # back to project root
python evaluate.py
```

---

## Complete Pipeline Summary

```
Raw data files (.h5 / .mtx.gz)
        │
        ▼
[Step 5] prepare_GSE152988.py     → GSE152988_processed.h5ad
[Step 6] prepare_GSE124703.py     → GSE124703_processed.h5ad
         (QC filter, normalize, extract perturbation labels from filenames/sgRNA files, DA scoring)
        │
        ▼
[Step 7] align_genes.py           → GSE152988_aligned.h5ad
                                    GSE124703_aligned.h5ad
         (Align to scFoundation 19,264-gene vocab; zero-pad missing genes)
        │
        ▼
[Step 8] split_data.py            → *_train.h5ad, *_val.h5ad, *_test.h5ad
         (Split by condition/guide, NOT by individual cell)
        │
        ▼
[Step 9] run_GSE152988.sh         → scFoundation extracts gene context embeddings
         run_GSE124703.sh            GEARS uses embeddings as graph node features
                                     Trains perturbation prediction model
                                     Saves predictions + train.log
        │
        ▼
[Step 10] evaluate.py             → Pearson (all), Pearson (top-20 DEG), MSE, R²
                                    metrics.json saved per dataset
```

---

## Known Issues and Workarounds

| Issue | Description | Fix |
|-------|-------------|-----|
| **Unfiltered barcodes** | 737,280 cells per GSE124703 sample is unfiltered Cell Ranger output | Run `sc.pp.filter_cells(min_genes=200)` — real count will be ~10k–50k per sample |
| **Missing sgRNA→gene map** | 57 guide barcodes in GSE124703 are raw sequences, not gene names | Get Supplementary Table S4 from Tian et al. 2019 (Neuron, PMID: 31422865) |
| **No per-cell perturbation labels** | GSE152988 has no per-cell labels — only file-level CRISPRi/CRISPRa | Use filename to assign condition label; true per-gene targets need the paper supplement |
| **Gene vocab mismatch** | scFoundation uses 19,264 genes; your datasets have ~33k genes | Step 7 handles this with zero-padding for missing genes |
| **Session crashes on large matrices** | Full matrix loading causes OOM | Use `sc.read_mtx()` then immediately filter; avoid loading all 4 matrices simultaneously |
| **GPU OOM during training** | Gene embeddings are 19264×512 per cell | Use `finetune_method=frozen` (keeps scFoundation frozen); reduce `batch_size` to 8 if needed |
| **DA neurons not confirmed in GSE124703** | TH=10.99 CPM, SLC6A3=0 CPM — generic neurons not DA | Use GSE152988 as primary DA dataset; GSE124703 is secondary/comparison |

---

## Important Project Context

- **DA reference dataset:** Use **GSE140231** (TA-recommended) to define DA neuron identity labels before training
- **Model selection rationale:** scFoundation is rejected for scGPT-brain (pretraining not perturbation-specific). scFoundation+GEARS uses perturbation-specific downstream training, partially addressing this limitation
- **Benchmarking context:** Multiple 2024–2025 papers show simple baselines (mean expression, PCA) often match foundation models for perturbation prediction. Always report a mean-expression baseline alongside scFoundation results
- **Evaluation order:**
  1. Preprocess GSE140231 → get clean DA labels
  2. Score cells in GSE152988 using DA markers → assign DA identity
  3. Train scFoundation+GEARS on perturbation task
  4. Report perturbation metrics separately for DA-high vs DA-low cells
- **Fine-tuning modes:**
  - `frozen` — scFoundation weights fixed; only GEARS trained (lower memory, faster)
  - `finetune_lr_1` — both scFoundation and GEARS fine-tuned (higher memory, potentially better)
  - Start with `frozen` due to GPU memory constraints
- **sgRNA guide counts:**
  - GSE152988: 184 CRISPRi + 100 CRISPRa gene targets
  - GSE124703: 57 guides → ~20–25 gene targets (same guides in iPSC + neuron contexts)

---

## File Structure After Setup

```
project/
├── scFoundation/              # cloned repo
│   ├── model/models/models.ckpt
│   └── GEARS/
│       ├── main.py
│       └── run_sh/
│           ├── run_GSE152988.sh
│           └── run_GSE124703.sh
├── data/
│   ├── DA_gene_dataset.csv
│   ├── GSE152988/             # 6 .h5 files
│   ├── GSE124703/             # MTX triplets + sgRNA files
│   ├── GSE152988_processed.h5ad
│   ├── GSE152988_aligned.h5ad
│   ├── GSE152988_train.h5ad
│   ├── GSE152988_val.h5ad
│   ├── GSE152988_test.h5ad
│   ├── GSE124703_processed.h5ad
│   ├── GSE124703_aligned.h5ad
│   ├── GSE124703_train.h5ad
│   ├── GSE124703_val.h5ad
│   └── GSE124703_test.h5ad
├── results/
│   ├── GSE152988/train.log
│   ├── GSE152988/metrics.json
│   ├── GSE124703/train.log
│   └── GSE124703/metrics.json
├── prepare_GSE152988.py
├── prepare_GSE124703.py
├── align_genes.py
├── split_data.py
└── evaluate.py
```
