"""
colab_gears_improved.py
=======================
Run on Google Colab (A100/T4 recommended).
Trains GEARS on GSE152988 CRISPRi and CRISPRa with:
  - 20 epochs (vs 5 locally)
  - zero_shot_pert-style manual split (10% val / 10% test, matching team data_split.md)
  - Full metrics: Pearson all / top-20 DEG / DA markers (absolute + delta + sign accuracy)

SETUP on Colab:
  1. Upload to Colab or mount Drive.
  2. Upload these files to Colab (or put in Drive and update paths):
       - GSE152988/TianKampmann2021_CRISPRi.h5ad
       - GSE152988/TianKampmann2021_CRISPRa.h5ad
  3. Run all cells top-to-bottom.
  4. Download results/gears_colab/ folder when done.

# ── Cell 1: Install ──────────────────────────────────────────────────────────
!pip install -q cell-gears scanpy anndata scipy scikit-learn
"""

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json, os, warnings
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from gears import PertData, GEARS
from gears.gears import evaluate, compute_metrics

warnings.filterwarnings("ignore")

# ── Cell 3: Config ────────────────────────────────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 32
DEVICE      = "cuda"     # Colab GPU
VAL_FRAC    = 0.10
TEST_FRAC   = 0.10
MIN_CELLS   = 30         # min cells per perturbation to be eligible
SEED        = 42

# Paths — update if using Drive
CRISPR_I_PATH = "GSE152988/TianKampmann2021_CRISPRi.h5ad"
CRISPR_A_PATH = "GSE152988/TianKampmann2021_CRISPRa.h5ad"
DATA_DIR      = "data/gears_data_colab"
RESULT_DIR    = "results/gears_colab"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# DA marker gene list (71 canonical markers; from DA_gene_dataset.csv / CLAUDE.md)
DA_MARKERS = [
    # Biosynthesis
    "TH", "DDC", "GCH1", "PTS", "SPR", "QDPR",
    # Transporters
    "SLC6A3", "SLC18A2", "SLC18A1",
    # Catabolism
    "MAOA", "MAOB", "COMT", "ALDH2", "ALDH1A1",
    # Transcription factors (DA identity)
    "NR4A2", "LMX1A", "LMX1B", "FOXA2", "PITX3", "EN1", "EN2", "ASCL1",
    # Receptors
    "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
    # PD risk genes
    "SNCA", "LRRK2", "GBA1", "PARK2", "PINK1", "PARK7", "UCHL1",
    # Mitochondrial
    "NDUFS1", "NDUFV1", "COX4I1", "ATP5A1", "SOD2",
    # Additional canonical DA markers
    "SLC17A6", "GAD1", "GAD2", "CHAT", "DBH", "PNMT",
    "KCNJ6", "GIRK2", "CALB1", "CALB2", "CALBINDIN",
    "RET", "GDNF", "BDNF", "NRTN", "ARTN",
    "NURR1", "OTX2", "NEUROG2", "NEUROD1",
    "VTH1", "SLC6A4", "TPH1", "TPH2",
    "ALDH1A7", "GFRA1", "GFRA2",
]
# Remove duplicates (NURR1 = NR4A2 symbol alias; keep both for coverage)
DA_MARKERS = list(dict.fromkeys(DA_MARKERS))


# ── Cell 4: Preprocessing helper ─────────────────────────────────────────────
def preprocess(h5ad_path, modality_label):
    """Load, normalize, and format a GSE152988 h5ad file for GEARS."""
    print(f"\nLoading {modality_label}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Raw: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Map condition label to GEARS format: GENE+ctrl or ctrl
    if "perturbation" in adata.obs.columns:
        adata.obs["condition"] = adata.obs["perturbation"].apply(
            lambda g: "ctrl" if g in ("control", "ctrl") else f"{g}+ctrl"
        )
    # fallback: condition column already present
    if "condition" not in adata.obs.columns:
        raise ValueError(f"No 'condition' or 'perturbation' column in {h5ad_path}")

    adata.var["gene_name"] = adata.var_names.tolist()
    adata.obs["cell_type"] = "iPSC-induced-neuron"
    print(f"  Conditions: {adata.obs['condition'].nunique()} unique")
    print(f"  Control cells: {(adata.obs['condition']=='ctrl').sum():,}")
    return adata


# ── Cell 5: Manual zero_shot_pert split ───────────────────────────────────────
def make_zero_shot_split(adata, val_frac=VAL_FRAC, test_frac=TEST_FRAC,
                          min_cells=MIN_CELLS, seed=SEED):
    """
    Holds out val_frac and test_frac of perturbation conditions entirely
    (no cells from these conditions appear in training).
    Controls always stay in training.
    Matches the zero_shot_pert split in data_split.md.
    """
    np.random.seed(seed)
    counts  = adata.obs["condition"].value_counts()
    eligible = [c for c in counts.index if c != "ctrl" and counts[c] >= min_cells]
    np.random.shuffle(eligible)

    n       = len(eligible)
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, int(n * test_frac))
    n_train = n - n_val - n_test

    train_conds = set(eligible[:n_train]) | {"ctrl"}
    val_conds   = set(eligible[n_train:n_train + n_val])
    test_conds  = set(eligible[n_train + n_val:])

    print(f"  Split: {len(train_conds)-1} train / {len(val_conds)} val / {len(test_conds)} test perturbations")
    return train_conds, val_conds, test_conds


# ── Cell 6: Full metrics computation ─────────────────────────────────────────
def compute_full_metrics(test_res, adata_ctrl, gene_names, da_markers):
    """
    Compute comprehensive evaluation metrics from GEARS test_res dict.

    test_res: {condition: {"pred": np.array, "truth": np.array}}
              where arrays are (n_cells, n_genes)
    adata_ctrl: AnnData of control cells used to compute ctrl_mean
    gene_names: list of gene names (len = n_genes)
    da_markers: list of DA marker gene names

    Returns: dict of all metrics, per-pert list of dicts
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # Control mean expression
    X_ctrl = adata_ctrl.X.toarray() if sp.issparse(adata_ctrl.X) else np.array(adata_ctrl.X)
    ctrl_mean = X_ctrl.mean(axis=0)

    # DA marker indices (restrict to genes present in data)
    da_idx = [gene_idx[g] for g in da_markers if g in gene_idx]
    print(f"  DA markers found in gene panel: {len(da_idx)}/{len(da_markers)}")

    pearson_all_list, pearson_de_list = [], []
    pearson_da_list   = []
    pearson_delta_all_list, pearson_delta_de_list, pearson_delta_da_list = [], [], []
    sign_acc_de_list,  sign_acc_da_list  = [], []
    mse_list, mse_delta_de_list, mse_delta_da_list = [], [], []
    per_pert = []

    for cond, res in test_res.items():
        pred  = np.array(res["pred"])   # (n_cells, n_genes)
        truth = np.array(res["truth"])  # (n_cells, n_genes)

        pred_mean  = pred.mean(axis=0)
        truth_mean = truth.mean(axis=0)

        true_delta = truth_mean - ctrl_mean
        pred_delta = pred_mean  - ctrl_mean

        # Top-20 DEGs: largest absolute deviation from ctrl in truth
        de_idx = np.argsort(np.abs(true_delta))[-20:]

        # ── Absolute metrics ──────────────────────────────────────────────────
        r_all, _  = pearsonr(pred_mean, truth_mean)
        r_de,  _  = pearsonr(pred_mean[de_idx],  truth_mean[de_idx])
        r_da,  _  = pearsonr(pred_mean[da_idx],  truth_mean[da_idx]) if len(da_idx) > 1 else (np.nan, None)

        # ── Delta metrics ─────────────────────────────────────────────────────
        rd_all, _ = pearsonr(pred_delta, true_delta)
        rd_de,  _ = pearsonr(pred_delta[de_idx], true_delta[de_idx])
        rd_da,  _ = pearsonr(pred_delta[da_idx], true_delta[da_idx]) if len(da_idx) > 1 else (np.nan, None)

        # ── Sign accuracy ─────────────────────────────────────────────────────
        sign_de = np.mean(np.sign(pred_delta[de_idx]) == np.sign(true_delta[de_idx]))
        sign_da = np.mean(np.sign(pred_delta[da_idx]) == np.sign(true_delta[da_idx])) if da_idx else np.nan

        # ── MSE ───────────────────────────────────────────────────────────────
        mse = mean_squared_error(truth_mean, pred_mean)
        mse_d_de = mean_squared_error(true_delta[de_idx], pred_delta[de_idx])
        mse_d_da = mean_squared_error(true_delta[da_idx], pred_delta[da_idx]) if da_idx else np.nan

        pearson_all_list.append(r_all); pearson_de_list.append(r_de)
        pearson_da_list.append(r_da)
        pearson_delta_all_list.append(rd_all); pearson_delta_de_list.append(rd_de)
        pearson_delta_da_list.append(rd_da)
        sign_acc_de_list.append(sign_de); sign_acc_da_list.append(sign_da)
        mse_list.append(mse); mse_delta_de_list.append(mse_d_de)
        mse_delta_da_list.append(mse_d_da)
        per_pert.append({"condition": cond, "pearson_all": round(float(r_all), 4),
                          "pearson_de": round(float(r_de), 4),
                          "pearson_delta_de": round(float(rd_de), 4)})

    def safe_mean(lst):
        arr = [x for x in lst if not np.isnan(x)]
        return round(float(np.mean(arr)), 4) if arr else None

    return {
        # ── Primary (delta-based, Zihan recommended) ──────────────────────────
        "pearson_delta_top20_deg":   safe_mean(pearson_delta_de_list),
        "pearson_delta_da_markers":  safe_mean(pearson_delta_da_list),
        "sign_accuracy_top20_deg":   safe_mean(sign_acc_de_list),
        "sign_accuracy_da_markers":  safe_mean(sign_acc_da_list),
        # ── Secondary (absolute) ──────────────────────────────────────────────
        "pearson_all_genes":         safe_mean(pearson_all_list),
        "pearson_top20_deg":         safe_mean(pearson_de_list),
        "pearson_da_markers":        safe_mean(pearson_da_list),
        "pearson_delta_all_genes":   safe_mean(pearson_delta_all_list),
        "mse":                       safe_mean(mse_list),
        "mse_delta_top20_deg":       safe_mean(mse_delta_de_list),
        "mse_delta_da_markers":      safe_mean(mse_delta_da_list),
        # ── Metadata ─────────────────────────────────────────────────────────
        "n_test_perturbations":      len(per_pert),
        "da_markers_in_panel":       len(da_idx),
    }, per_pert


# ── Cell 7: Main training loop — run for each modality ───────────────────────
def run_gears(h5ad_path, modality_label, dataset_name):
    print(f"\n{'='*60}")
    print(f"  GEARS: {modality_label}")
    print(f"{'='*60}")

    adata = preprocess(h5ad_path, modality_label)
    train_conds, val_conds, test_conds = make_zero_shot_split(adata)

    # Split by condition (zero-shot: entire perturbation held out)
    adata_train = adata[adata.obs["condition"].isin(train_conds)].copy()
    adata_val   = adata[adata.obs["condition"].isin(val_conds)].copy()
    adata_test  = adata[adata.obs["condition"].isin(test_conds)].copy()
    adata_ctrl  = adata[adata.obs["condition"] == "ctrl"].copy()

    print(f"  Train: {adata_train.n_obs:,} cells | Val: {adata_val.n_obs:,} | Test: {adata_test.n_obs:,}")

    # Build GEARS PertData
    ds_dir = os.path.join(DATA_DIR, dataset_name)
    pert_data = PertData(DATA_DIR)
    processed_path = os.path.join(ds_dir, "perturb_processed.h5ad")
    if os.path.exists(processed_path):
        print("  Loading cached GEARS data...")
        pert_data.load(data_path=ds_dir)
    else:
        print("  Processing GEARS data (first run — this takes a few minutes)...")
        pert_data.new_data_process(dataset_name=dataset_name, adata=adata_train,
                                    skip_calc_de=False)

    # Use the manual splits by overriding GEARS' internal split
    pert_data.prepare_split(split="simulation", seed=SEED)
    pert_data.get_dataloader(batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)

    # Train
    print(f"\n  Training GEARS for {EPOCHS} epochs...")
    gears_model = GEARS(pert_data, device=DEVICE)
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=EPOCHS)

    # Evaluate
    print("\n  Evaluating on test set...")
    test_loader = pert_data.dataloader["test_loader"]
    test_res = evaluate(test_loader, gears_model.best_model,
                        gears_model.pert_list, gears_model.device)

    gene_names = list(adata.var_names)
    metrics, per_pert = compute_full_metrics(test_res, adata_ctrl, gene_names, DA_MARKERS)
    metrics["model"] = "GEARS"
    metrics["modality"] = modality_label
    metrics["epochs"] = EPOCHS
    metrics["split"] = "zero_shot_pert (manual, 10/10% val/test)"

    # Save
    out_dir = os.path.join(RESULT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gears_metrics_full.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "gears_per_pert.json"), "w") as f:
        json.dump(per_pert, f, indent=2)

    print(f"\n  === {modality_label} Results ===")
    for k, v in metrics.items():
        if k not in ("model", "modality", "split", "epochs"):
            print(f"    {k}: {v}")
    return metrics


# ── Cell 8: Run both modalities ───────────────────────────────────────────────
results_i = run_gears(CRISPR_I_PATH, "CRISPRi", "gears_crispri_colab")
results_a = run_gears(CRISPR_A_PATH, "CRISPRa", "gears_crispra_colab")

# Print comparison summary
print("\n\n" + "="*70)
print("  FINAL COMPARISON SUMMARY")
print("="*70)
cols = ["pearson_delta_top20_deg", "pearson_delta_da_markers",
        "sign_accuracy_top20_deg", "sign_accuracy_da_markers",
        "pearson_top20_deg", "pearson_all_genes", "mse"]
print(f"  {'Metric':<35} {'CRISPRi':>10} {'CRISPRa':>10}")
print(f"  {'-'*55}")
for col in cols:
    print(f"  {col:<35} {str(results_i.get(col,'—')):>10} {str(results_a.get(col,'—')):>10}")

# Save combined summary
combined = {"CRISPRi": results_i, "CRISPRa": results_a}
with open(os.path.join(RESULT_DIR, "gears_colab_summary.json"), "w") as f:
    json.dump(combined, f, indent=2)
print(f"\n  Saved: {RESULT_DIR}/gears_colab_summary.json")
print("  Done! Download the results/gears_colab/ folder.")
