"""
colab_gears_improved.py
=======================
Run on Google Colab (A100/T4 recommended).
Trains GEARS on GSE152988 CRISPRi and CRISPRa with:
  - 20 epochs (vs 5 locally)
  - zero_shot_pert-style manual split (10% val / 10% test, matching team data_split.md)
  - Full metrics: Pearson all / top-20 DEG / DA markers (absolute + delta + sign accuracy)
  - CHECKPOINTING: survives Colab disconnects — resumes from last saved state automatically

═══════════════════════════════════════════════════════════════
SETUP (run these cells in order in Colab)
═══════════════════════════════════════════════════════════════

# ── Cell 1: Mount Google Drive (REQUIRED for checkpoints to persist) ──────────
from google.colab import drive
drive.mount('/content/drive')

# ── Cell 2: Install dependencies ──────────────────────────────────────────────
!pip install -q cell-gears scanpy anndata scipy scikit-learn

# ── Cell 3: Upload data files to Drive, then set paths below ─────────────────
# Put these two files anywhere in your Drive, e.g.:
#   My Drive/566/GSE152988/TianKampmann2021_CRISPRi.h5ad
#   My Drive/566/GSE152988/TianKampmann2021_CRISPRa.h5ad
# Then update DRIVE_BASE below to match.

# ── Cell 4: Run this script ───────────────────────────────────────────────────
# exec(open('colab_gears_improved.py').read())   # if uploaded as a file
# OR paste everything below into a cell and run it.
═══════════════════════════════════════════════════════════════
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json, os, shutil, warnings
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from gears import PertData, GEARS

warnings.filterwarnings("ignore")

# Compatibility fix: pandas >= 2.0 removed Series.nonzero() which GEARS still calls internally
if not hasattr(pd.Series, 'nonzero'):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — update DRIVE_BASE to your Google Drive folder
# ═══════════════════════════════════════════════════════════════════════════════
DRIVE_BASE = "/content/drive/MyDrive/566"   # <-- change this if your Drive path differs

CRISPR_I_PATH = f"{DRIVE_BASE}/GSE152988/TianKampmann2021_CRISPRi.h5ad"
CRISPR_A_PATH = f"{DRIVE_BASE}/GSE152988/TianKampmann2021_CRISPRa.h5ad"

# All checkpoints and results go to Drive so they survive disconnects
CHECKPOINT_DIR = f"{DRIVE_BASE}/gears_checkpoints"   # model weights + split info
RESULT_DIR     = f"{DRIVE_BASE}/results/gears_colab" # final + incremental results
DATA_DIR       = "/content/gears_data"               # GEARS graph cache (local, fast I/O)

EPOCHS      = 20
BATCH_SIZE  = 32
DEVICE      = "cuda"
VAL_FRAC    = 0.10
TEST_FRAC   = 0.10
MIN_CELLS   = 30
SEED        = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# DA marker genes (from project CLAUDE.md)
DA_MARKERS = [
    "TH", "DDC", "GCH1", "PTS", "SPR", "QDPR",
    "SLC6A3", "SLC18A2", "SLC18A1",
    "MAOA", "MAOB", "COMT", "ALDH2", "ALDH1A1",
    "NR4A2", "LMX1A", "LMX1B", "FOXA2", "PITX3", "EN1", "EN2", "ASCL1",
    "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
    "SNCA", "LRRK2", "GBA1", "PARK2", "PINK1", "PARK7", "UCHL1",
    "NDUFS1", "NDUFV1", "COX4I1", "ATP5A1", "SOD2",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def ckpt_path(dataset_name, fname):
    d = os.path.join(CHECKPOINT_DIR, dataset_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)

def save_model(gears_model, dataset_name):
    """Save model weights + pert_list to Drive checkpoint."""
    path = ckpt_path(dataset_name, "best_model.pt")
    torch.save({
        "model_state": gears_model.best_model.state_dict(),
        "pert_list":   gears_model.pert_list,
    }, path)
    print(f"  [ckpt] Model saved -> {path}")

def load_model_if_exists(gears_model, dataset_name):
    """Load model weights from Drive if checkpoint exists. Returns True if loaded."""
    path = ckpt_path(dataset_name, "best_model.pt")
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, map_location=gears_model.device)
    gears_model.best_model.load_state_dict(ckpt["model_state"])
    gears_model.best_model.eval()
    print(f"  [ckpt] Loaded model from {path} — skipping training.")
    return True

def save_split(train_conds, val_conds, test_conds, dataset_name):
    """Persist the condition split so restarts use identical splits."""
    data = {
        "train": sorted(train_conds),
        "val":   sorted(val_conds),
        "test":  sorted(test_conds),
    }
    with open(ckpt_path(dataset_name, "split.json"), "w") as f:
        json.dump(data, f, indent=2)

def load_split_if_exists(dataset_name):
    """Load split from Drive if it was already computed. Returns (train, val, test) or None."""
    path = ckpt_path(dataset_name, "split.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  [ckpt] Loaded split from {path}")
    return set(data["train"]), set(data["val"]), set(data["test"])

def load_partial_results(dataset_name):
    """Load per-perturbation results already evaluated in a previous run."""
    path = os.path.join(RESULT_DIR, dataset_name, "per_pert_partial.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        records = json.load(f)
    done = {r["condition"]: r for r in records}
    print(f"  [ckpt] Resuming: {len(done)} perturbations already evaluated.")
    return done

def save_partial_results(per_pert_list, dataset_name):
    """Append-safe incremental save of per-perturbation results to Drive."""
    out_dir = os.path.join(RESULT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_pert_partial.json"), "w") as f:
        json.dump(per_pert_list, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess(h5ad_path, modality_label):
    print(f"\nLoading {modality_label} from {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Raw: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if "perturbation" in adata.obs.columns:
        adata.obs["condition"] = adata.obs["perturbation"].apply(
            lambda g: "ctrl" if g in ("control", "ctrl") else f"{g}+ctrl"
        )
    if "condition" not in adata.obs.columns:
        raise ValueError(f"No 'condition' or 'perturbation' column found in {h5ad_path}")
    adata.var["gene_name"] = adata.var_names.tolist()
    adata.obs["cell_type"] = "iPSC-induced-neuron"
    print(f"  Conditions: {adata.obs['condition'].nunique()} unique  |  "
          f"Control cells: {(adata.obs['condition']=='ctrl').sum():,}")
    return adata


# ═══════════════════════════════════════════════════════════════════════════════
#  SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
def make_zero_shot_split(adata, dataset_name):
    """zero_shot_pert split — deterministic. Loads from checkpoint if already saved."""
    existing = load_split_if_exists(dataset_name)
    if existing:
        return existing

    np.random.seed(SEED)
    counts   = adata.obs["condition"].value_counts()
    eligible = [c for c in counts.index if c != "ctrl" and counts[c] >= MIN_CELLS]
    np.random.shuffle(eligible)

    n       = len(eligible)
    n_val   = max(1, int(n * VAL_FRAC))
    n_test  = max(1, int(n * TEST_FRAC))
    n_train = n - n_val - n_test

    train_conds = set(eligible[:n_train]) | {"ctrl"}
    val_conds   = set(eligible[n_train:n_train + n_val])
    test_conds  = set(eligible[n_train + n_val:])

    save_split(train_conds, val_conds, test_conds, dataset_name)
    print(f"  Split: {len(train_conds)-1} train / {len(val_conds)} val / "
          f"{len(test_conds)} test perturbations")
    return train_conds, val_conds, test_conds


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_full_metrics(test_res, ctrl_mean, gene_names, da_markers, dataset_name):
    """
    Compute all metrics. Resumes from partial results if a previous run saved them.
    Saves to Drive after each perturbation so any crash only loses the current one.

    ctrl_mean: 1D np.array of shape (n_genes,) in the SAME normalized space as batch.y.
               Must be computed from the GEARS dataloader, not from adata.X directly.
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    da_idx   = [gene_idx[g] for g in da_markers if g in gene_idx]
    print(f"  DA markers found in gene panel: {len(da_idx)}/{len(da_markers)}")

    # Resume from partial results if available
    done       = load_partial_results(dataset_name)
    per_pert   = list(done.values())   # already-evaluated records

    todo = {k: v for k, v in test_res.items() if k not in done}
    print(f"  Evaluating {len(todo)} perturbations "
          f"({len(done)} already done from previous run)...")

    for cond, res in todo.items():
        pred  = np.array(res["pred"])
        truth = np.array(res["truth"])

        pred_mean  = pred.mean(axis=0)
        truth_mean = truth.mean(axis=0)
        true_delta = truth_mean - ctrl_mean
        pred_delta = pred_mean  - ctrl_mean
        de_idx     = np.argsort(np.abs(true_delta))[-20:]

        def safe_r(a, b):
            if len(a) < 2: return None
            r, _ = pearsonr(a, b)
            return round(float(r), 4)

        record = {
            "condition":             cond,
            "pearson_delta_de":      safe_r(pred_delta[de_idx], true_delta[de_idx]),
            "pearson_delta_da":      safe_r(pred_delta[da_idx], true_delta[da_idx]) if da_idx else None,
            "sign_accuracy_de":      round(float(np.mean(np.sign(pred_delta[de_idx]) == np.sign(true_delta[de_idx]))), 4),
            "sign_accuracy_da":      round(float(np.mean(np.sign(pred_delta[da_idx]) == np.sign(true_delta[da_idx]))), 4) if da_idx else None,
            "pearson_all":           safe_r(pred_mean, truth_mean),
            "pearson_de":            safe_r(pred_mean[de_idx], truth_mean[de_idx]),
            "pearson_da":            safe_r(pred_mean[da_idx], truth_mean[da_idx]) if da_idx else None,
            "pearson_delta_all":     safe_r(pred_delta, true_delta),
            "mse":                   round(float(mean_squared_error(truth_mean, pred_mean)), 6),
            "mse_delta_de":          round(float(mean_squared_error(true_delta[de_idx], pred_delta[de_idx])), 6),
            "mse_delta_da":          round(float(mean_squared_error(true_delta[da_idx], pred_delta[da_idx])), 6) if da_idx else None,
            "n_da_in_panel":         len(da_idx),
        }
        per_pert.append(record)
        # Save to Drive after every single perturbation
        save_partial_results(per_pert, dataset_name)

    # Aggregate
    def safe_mean(key):
        vals = [r[key] for r in per_pert if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    keys = ["pearson_delta_de", "pearson_delta_da", "sign_accuracy_de", "sign_accuracy_da",
            "pearson_all", "pearson_de", "pearson_da", "pearson_delta_all",
            "mse", "mse_delta_de", "mse_delta_da"]
    agg = {k + "_mean": safe_mean(k) for k in keys}
    agg["n_test_perturbations"] = len(per_pert)
    agg["da_markers_in_panel"]  = len(da_idx)
    return agg, per_pert


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM EVALUATE — bypasses GEARS' internal evaluate() API entirely
#  Works regardless of GEARS version or uncertainty flag behaviour
# ═══════════════════════════════════════════════════════════════════════════════
def custom_evaluate(test_loader, model, device):
    """
    Run model on test_loader and collect per-perturbation predictions and ground truth.
    Returns: {condition_str: {"pred": np.array (n_cells, n_genes),
                              "truth": np.array (n_cells, n_genes)}}
    """
    model.eval()
    results = {}

    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
            # Handle tuple output (some GEARS versions return (pred, uncertainty))
            pred = output[0] if isinstance(output, (tuple, list)) else output

        preds  = pred.cpu().numpy()   # (batch_size, n_genes)
        truths = batch.y.cpu().numpy()  # (batch_size, n_genes)

        # pert is a list of strings like ["GENE+ctrl", "GENE+ctrl", ...]
        for i, pert in enumerate(batch.pert):
            if pert == "ctrl":
                continue
            if pert not in results:
                results[pert] = {"pred": [], "truth": []}
            results[pert]["pred"].append(preds[i])
            results[pert]["truth"].append(truths[i])

    # Stack into arrays
    for pert in results:
        results[pert]["pred"]  = np.stack(results[pert]["pred"])
        results[pert]["truth"] = np.stack(results[pert]["truth"])

    print(f"  Evaluated {len(results)} test perturbations.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING + EVAL LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def run_gears(h5ad_path, modality_label, dataset_name):
    print(f"\n{'='*65}")
    print(f"  GEARS: {modality_label}  |  dataset: {dataset_name}")
    print(f"{'='*65}")

    # ── Check if final results already exist (full run complete) ──────────────
    final_path = os.path.join(RESULT_DIR, dataset_name, "gears_metrics_full.json")
    partial_path = os.path.join(RESULT_DIR, dataset_name, "per_pert_partial.json")
    if os.path.exists(final_path):
        print(f"  [ckpt] Final results already exist at {final_path} — skipping.")
        with open(final_path) as f:
            return json.load(f)
    # Clear stale partial results so evaluation reruns with correct ctrl_mean
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print("  [ckpt] Cleared stale partial eval results — will recompute with fixed ctrl_mean.")

    adata = preprocess(h5ad_path, modality_label)
    train_conds, val_conds, test_conds = make_zero_shot_split(adata, dataset_name)

    adata_train = adata[adata.obs["condition"].isin(train_conds)].copy()
    adata_ctrl  = adata[adata.obs["condition"] == "ctrl"].copy()
    print(f"  Train: {adata_train.n_obs:,} cells  |  "
          f"Test conditions: {len(test_conds)}")

    # ── Build PertData (cached locally; graph download only needed once) ──────
    ds_dir = os.path.join(DATA_DIR, dataset_name)

    # Also cache PertData to Drive so it survives full session resets
    drive_ds_dir = ckpt_path(dataset_name, "pert_data")
    local_processed = os.path.join(ds_dir, "perturb_processed.h5ad")
    drive_processed = os.path.join(drive_ds_dir, "perturb_processed.h5ad")

    pert_data = PertData(DATA_DIR)
    if os.path.exists(local_processed):
        print("  [ckpt] Loading GEARS data from local cache...")
        pert_data.load(data_path=ds_dir)
    elif os.path.exists(drive_processed):
        print("  [ckpt] Copying GEARS data from Drive cache to local...")
        shutil.copytree(drive_ds_dir, ds_dir, dirs_exist_ok=True)
        pert_data.load(data_path=ds_dir)
    else:
        print("  Processing GEARS data (first run — ~5 min)...")
        pert_data.new_data_process(dataset_name=dataset_name, adata=adata_train,
                                    skip_calc_de=False)
        # Back up to Drive immediately
        print("  [ckpt] Backing up GEARS data to Drive...")
        shutil.copytree(ds_dir, drive_ds_dir, dirs_exist_ok=True)

    pert_data.prepare_split(split="simulation", seed=SEED)
    pert_data.get_dataloader(batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)

    # ── Train (skip if checkpoint exists) ────────────────────────────────────
    gears_model = GEARS(pert_data, device=DEVICE)
    gears_model.model_initialize(hidden_size=64)

    trained_from_ckpt = load_model_if_exists(gears_model, dataset_name)
    if not trained_from_ckpt:
        print(f"\n  Training GEARS for {EPOCHS} epochs...")
        gears_model.train(epochs=EPOCHS)
        save_model(gears_model, dataset_name)
        print("  Training complete. Checkpoint saved to Drive.")
    else:
        # best_model needs to be set for evaluate() to work
        gears_model.best_model = gears_model.model

    # ── Compute ctrl_mean from GEARS' own dataloader (same normalized space as batch.y) ──
    print("\n  Computing ctrl_mean from GEARS train_loader...")
    ctrl_exprs = []
    for batch in pert_data.dataloader["train_loader"]:
        batch = batch.to(gears_model.device)
        for i, pert_name in enumerate(batch.pert):
            if pert_name == "ctrl":
                ctrl_exprs.append(batch.y[i].cpu().numpy())
    if not ctrl_exprs:
        raise RuntimeError("No control cells found in train_loader.")
    ctrl_mean_gears = np.stack(ctrl_exprs).mean(axis=0)
    print(f"  ctrl_mean computed from {len(ctrl_exprs)} control cells in GEARS space.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n  Evaluating test set...")
    test_loader = pert_data.dataloader["test_loader"]
    test_res    = custom_evaluate(test_loader, gears_model.best_model, gears_model.device)

    gene_names = list(adata.var_names)
    metrics, per_pert = compute_full_metrics(
        test_res, ctrl_mean_gears, gene_names, DA_MARKERS, dataset_name
    )
    metrics.update({
        "model":    "GEARS",
        "modality": modality_label,
        "epochs":   EPOCHS,
        "split":    "zero_shot_pert (manual, 10/10% val/test)",
    })

    # ── Save final results to Drive ───────────────────────────────────────────
    out_dir = os.path.join(RESULT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(final_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "gears_per_pert.json"), "w") as f:
        json.dump(per_pert, f, indent=2)
    print(f"\n  [ckpt] Final results saved to Drive: {final_path}")

    print(f"\n  === {modality_label} Results ===")
    for k, v in metrics.items():
        if k not in ("model", "modality", "split", "epochs"):
            print(f"    {k}: {v}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════
results_i = run_gears(CRISPR_I_PATH, "CRISPRi", "gears_crispri_colab")
results_a = run_gears(CRISPR_A_PATH, "CRISPRa", "gears_crispra_colab")

# Summary
print("\n\n" + "="*70)
print("  FINAL COMPARISON SUMMARY")
print("="*70)
cols = ["pearson_delta_de_mean", "pearson_delta_da_mean",
        "sign_accuracy_de_mean", "sign_accuracy_da_mean",
        "pearson_de_mean", "pearson_all_mean", "mse_mean"]
print(f"  {'Metric':<35} {'CRISPRi':>10} {'CRISPRa':>10}")
print(f"  {'-'*57}")
for col in cols:
    print(f"  {col:<35} {str(results_i.get(col,'—')):>10} {str(results_a.get(col,'—')):>10}")

with open(os.path.join(RESULT_DIR, "gears_colab_summary.json"), "w") as f:
    json.dump({"CRISPRi": results_i, "CRISPRa": results_a}, f, indent=2)
print(f"\n  Summary saved: {RESULT_DIR}/gears_colab_summary.json")
print("  All done. Download the results from Google Drive.")
