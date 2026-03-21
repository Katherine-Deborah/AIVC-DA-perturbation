"""
04_mean_baseline.py
Mean expression baseline: predict each perturbation's effect as the mean
of training perturbed cells. A simple but often competitive baseline.
Outputs: results/baseline/baseline_metrics.json
"""
import numpy as np
import json
import os
import anndata as ad
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

INPUT      = "data/crispr_i_processed.h5ad"
RESULT_DIR = "results/baseline"
N_TEST_PERTS = 20

os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading data...")
adata = ad.read_h5ad(INPUT)

# Same split as scGen
np.random.seed(42)
all_perts = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
np.random.shuffle(all_perts)
test_perts  = set(all_perts[:N_TEST_PERTS])
train_perts = set(all_perts[N_TEST_PERTS:])

adata_train = adata[adata.obs["condition"].isin(train_perts)].copy()
adata_test  = adata[adata.obs["condition"].isin(test_perts)].copy()
adata_ctrl  = adata[adata.obs["condition"] == "ctrl"].copy()

# Mean of all training perturbed cells
X_train = adata_train.X
if sp.issparse(X_train):
    X_train = X_train.toarray()
global_mean_perturbed = X_train.mean(axis=0)

pearson_all_list, pearson_top20_list, mse_list, r2_list = [], [], [], []

for pert in sorted(test_perts):
    true_cells = adata_test[adata_test.obs["condition"] == pert]
    if true_cells.n_obs == 0:
        continue

    X_true = true_cells.X
    if sp.issparse(X_true):
        X_true = X_true.toarray()
    true_mean = X_true.mean(axis=0)

    pred_mean = global_mean_perturbed

    r_all, _ = pearsonr(pred_mean, true_mean)
    pearson_all_list.append(r_all)

    top20_idx = np.argsort(true_mean)[-20:]
    r_top, _ = pearsonr(pred_mean[top20_idx], true_mean[top20_idx])
    pearson_top20_list.append(r_top)

    mse = mean_squared_error(true_mean, pred_mean)
    mse_list.append(mse)

    ss_res = np.sum((true_mean - pred_mean) ** 2)
    ss_tot = np.sum((true_mean - true_mean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_list.append(r2)

metrics = {
    "pearson_all_mean":     round(float(np.mean(pearson_all_list)), 4),
    "pearson_top20_mean":   round(float(np.mean(pearson_top20_list)), 4),
    "mse_mean":             round(float(np.mean(mse_list)), 4),
    "r2_mean":              round(float(np.mean(r2_list)), 4),
    "n_test_perturbations": len(pearson_all_list),
}

with open(os.path.join(RESULT_DIR, "baseline_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("=== Mean Baseline Results ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print(f"\nSaved to {RESULT_DIR}/baseline_metrics.json")
