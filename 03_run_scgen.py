"""
03_run_scgen.py
Train and evaluate scGen on GSE152988 CRISPRi data.
scGen learns a VAE then shifts latent vectors to predict perturbation effects.
Outputs: results/scgen/scgen_metrics.json
"""
import numpy as np
import pandas as pd
import json
import os
import anndata as ad
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import scgen

INPUT       = "data/crispr_i_processed.h5ad"
RESULT_DIR  = "results/scgen"
MODEL_DIR   = "results/scgen/model"
EPOCHS      = 100
BATCH_SIZE  = 32
# Evaluate on 20 seen perturbations (hold out 20% of cells per perturbation)
EVAL_PERTS  = 20
TEST_FRAC   = 0.2   # fraction of cells per perturbation withheld for testing

os.makedirs(RESULT_DIR, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading processed data...")
adata = ad.read_h5ad(INPUT)
print(f"  Shape: {adata.shape}")

# ── 2. Select perturbations to evaluate; hold out 20% of their cells ──────────
# scGen is designed for SEEN perturbation evaluation:
# train on 80% of ctrl+perturbed cells, test on the withheld 20%.
np.random.seed(42)
all_perts = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
# Pick perturbations with enough cells for reliable evaluation
counts = adata.obs["condition"].value_counts()
valid_perts = [p for p in all_perts if counts.get(p, 0) >= 20]
eval_perts = sorted(np.random.choice(valid_perts, size=EVAL_PERTS, replace=False))
print(f"\n  Eval perturbations: {eval_perts[:5]} ...")

# Build train set: ctrl + 80% of each perturbation's cells
train_idx, test_idx = [], []
ctrl_idx = adata.obs.index[adata.obs["condition"] == "ctrl"].tolist()
train_idx.extend(ctrl_idx)

for pert in eval_perts:
    idx = adata.obs.index[adata.obs["condition"] == pert].tolist()
    n_test = max(1, int(len(idx) * TEST_FRAC))
    np.random.shuffle(idx)
    test_idx.extend(idx[:n_test])
    train_idx.extend(idx[n_test:])

# Include all other perturbation cells in training
other_perts = [c for c in all_perts if c not in eval_perts]
for pert in other_perts:
    idx = adata.obs.index[adata.obs["condition"] == pert].tolist()
    train_idx.extend(idx)

adata_train = adata[train_idx].copy()
adata_test  = adata[test_idx].copy()

print(f"  Train: {adata_train.n_obs:,} cells")
print(f"  Test:  {adata_test.n_obs:,} cells across {EVAL_PERTS} perturbations")

# ── 3. Set up scGen columns ───────────────────────────────────────────────────
# scGen expects 'perturbation' (ctrl vs gene) and 'cell_type' in obs
for a in [adata_train, adata_test]:
    a.obs["perturbation"] = a.obs["condition"].apply(
        lambda c: "ctrl" if c == "ctrl" else c.replace("+ctrl", "")
    )
    a.obs["cell_type"] = "iPSC-induced-neuron"

# ── 4. Train scGen ────────────────────────────────────────────────────────────
print("\nSetting up scGen model...")
scgen.SCGEN.setup_anndata(adata_train,
                           batch_key="perturbation",
                           labels_key="cell_type")

model = scgen.SCGEN(adata_train)
print("Training scGen...")
model.train(
    max_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping=True,
    early_stopping_patience=10,
)
model.save(MODEL_DIR, overwrite=True)
print(f"  Model saved to {MODEL_DIR}")

# ── 5. Evaluate on held-out cells of seen perturbations ───────────────────────
print("\nEvaluating on held-out cells of seen perturbations...")

import scipy.sparse as sp
import torch

pearson_all_list, pearson_top20_list, mse_list, r2_list = [], [], [], []

def encode(model, adata_subset):
    """Encode cells directly via the module's z_encoder — bypasses API mismatch."""
    import torch, scipy.sparse as sp
    X = adata_subset.X
    if sp.issparse(X):
        X = X.toarray()
    x_tensor = torch.tensor(X, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        _, _, z = model.module.z_encoder(x_tensor)
    return z.cpu().numpy()

# Precompute latent for all ctrl cells in training
ctrl_train = adata_train[adata_train.obs["perturbation"] == "ctrl"].copy()
z_ctrl_train = encode(model, ctrl_train)                            # (n_ctrl, n_latent)
z_ctrl_mean = z_ctrl_train.mean(axis=0)

# Precompute ctrl mean expression for DEG scoring
ctrl_mean_expr = (ctrl_train.X.toarray() if sp.issparse(ctrl_train.X)
                  else np.array(ctrl_train.X)).mean(axis=0)

for pert in eval_perts:
    pert_name = pert.replace("+ctrl", "")

    # True mean of withheld test cells
    true_cells = adata_test[adata_test.obs["condition"] == pert]
    if true_cells.n_obs == 0:
        continue
    true_mean = (true_cells.X.toarray() if sp.issparse(true_cells.X)
                 else np.array(true_cells.X)).mean(axis=0)

    # Get latent of perturbed cells in training set
    pert_train = adata_train[adata_train.obs["perturbation"] == pert_name].copy()
    if pert_train.n_obs == 0:
        print(f"  Skipping {pert}: no training cells found")
        continue

    try:
        z_pert_train = encode(model, pert_train)
        z_pert_mean = z_pert_train.mean(axis=0)

        # Compute delta vector in latent space
        delta = z_pert_mean - z_ctrl_mean

        # Apply delta to all ctrl training cells, then decode
        z_pred = z_ctrl_train + delta                              # (n_ctrl, n_latent)
        z_tensor = torch.tensor(z_pred, dtype=torch.float32)
        with torch.no_grad():
            pred_expr = model.module.generative(z_tensor)["px"].cpu().numpy()
        pred_mean = pred_expr.mean(axis=0)

    except Exception as e:
        print(f"  Skipping {pert}: {e}")
        continue

    r_all, _ = pearsonr(pred_mean, true_mean)
    pearson_all_list.append(r_all)

    # Top-20 DEGs: genes most changed vs ctrl
    delta_expr = np.abs(true_mean - ctrl_mean_expr)
    top20_idx = np.argsort(delta_expr)[-20:]
    r_top, _ = pearsonr(pred_mean[top20_idx], true_mean[top20_idx])
    pearson_top20_list.append(r_top)

    mse = mean_squared_error(true_mean, pred_mean)
    mse_list.append(mse)

    ss_res = np.sum((true_mean - pred_mean) ** 2)
    ss_tot = np.sum((true_mean - true_mean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_list.append(r2)
    print(f"  {pert_name}: Pearson={r_all:.3f}, top20={r_top:.3f}")

metrics = {
    "pearson_all_mean":    round(float(np.mean(pearson_all_list)), 4),
    "pearson_top20_mean":  round(float(np.mean(pearson_top20_list)), 4),
    "mse_mean":            round(float(np.mean(mse_list)), 4),
    "r2_mean":             round(float(np.mean(r2_list)), 4),
    "n_test_perturbations": len(pearson_all_list),
}

with open(os.path.join(RESULT_DIR, "scgen_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== scGen Results ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print(f"\nMetrics saved to {RESULT_DIR}/scgen_metrics.json")
print("scGen complete.")
