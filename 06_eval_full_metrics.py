"""
06_eval_full_metrics.py
=======================
Re-evaluates existing scGen and mean baseline models with the full metric suite
recommended in team resources/2026-03-21.md:

  Primary (delta-based):
    - pearson_delta_top20_deg
    - pearson_delta_da_markers
    - sign_accuracy_top20_deg
    - sign_accuracy_da_markers

  Secondary (absolute):
    - pearson_all_genes
    - pearson_top20_deg
    - pearson_da_markers
    - mse

Runs WITHOUT retraining — loads saved scGen model weights from results/scgen/model/.
Outputs updated metrics JSONs alongside existing ones.

Run locally: python 06_eval_full_metrics.py
"""

import numpy as np
import json, os, warnings
import anndata as ad
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import torch

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
EVAL_PERTS  = 20
TEST_FRAC   = 0.2

DATASETS = [
    {
        "label":      "CRISPRi",
        "input":      "data/crispr_i_processed.h5ad",
        "model_dir":  "results/scgen/model",
        "out_scgen":  "results/scgen/scgen_metrics_full.json",
        "out_base":   "results/baseline/baseline_metrics_full.json",
    },
    {
        "label":      "CRISPRa",
        "input":      "data/crispr_a_processed.h5ad",
        "model_dir":  "results/scgen_crispra/model",
        "out_scgen":  "results/scgen_crispra/scgen_metrics_full.json",
        "out_base":   "results/baseline_crispra/baseline_metrics_full.json",
    },
]

# DA marker list (from project CLAUDE.md)
DA_MARKERS = [
    "TH", "DDC", "GCH1", "PTS", "SPR", "QDPR",
    "SLC6A3", "SLC18A2", "SLC18A1",
    "MAOA", "MAOB", "COMT", "ALDH2", "ALDH1A1",
    "NR4A2", "LMX1A", "LMX1B", "FOXA2", "PITX3", "EN1", "EN2", "ASCL1",
    "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
    "SNCA", "LRRK2", "GBA1", "PARK2", "PINK1", "PARK7", "UCHL1",
    "NDUFS1", "NDUFV1", "COX4I1", "ATP5A1", "SOD2",
]


# ── Metric helpers ────────────────────────────────────────────────────────────
def full_metrics_for_condition(pred_mean, truth_mean, ctrl_mean, gene_names,
                                da_markers):
    """Compute all metrics for one perturbation condition."""
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    da_idx   = [gene_idx[g] for g in da_markers if g in gene_idx]

    true_delta = truth_mean - ctrl_mean
    pred_delta = pred_mean  - ctrl_mean
    de_idx     = np.argsort(np.abs(true_delta))[-20:]

    def safe_r(a, b):
        if len(a) < 2: return np.nan
        r, _ = pearsonr(a, b)
        return float(r)

    return {
        "pearson_all":           safe_r(pred_mean,        truth_mean),
        "pearson_de":            safe_r(pred_mean[de_idx], truth_mean[de_idx]),
        "pearson_da":            safe_r(pred_mean[da_idx], truth_mean[da_idx]),
        "pearson_delta_all":     safe_r(pred_delta,        true_delta),
        "pearson_delta_de":      safe_r(pred_delta[de_idx], true_delta[de_idx]),
        "pearson_delta_da":      safe_r(pred_delta[da_idx], true_delta[da_idx]) if da_idx else np.nan,
        "sign_accuracy_de":      float(np.mean(np.sign(pred_delta[de_idx]) == np.sign(true_delta[de_idx]))),
        "sign_accuracy_da":      float(np.mean(np.sign(pred_delta[da_idx]) == np.sign(true_delta[da_idx]))) if da_idx else np.nan,
        "mse":                   float(mean_squared_error(truth_mean, pred_mean)),
        "mse_delta_de":          float(mean_squared_error(true_delta[de_idx], pred_delta[de_idx])),
        "mse_delta_da":          float(mean_squared_error(true_delta[da_idx], pred_delta[da_idx])) if da_idx else np.nan,
        "n_da_in_panel":         len(da_idx),
    }


def aggregate(per_pert_list):
    """Average per-perturbation metrics, skipping NaNs."""
    keys = [k for k in per_pert_list[0] if k != "n_da_in_panel"]
    agg = {}
    for k in keys:
        vals = []
        for d in per_pert_list:
            v = d[k]
            try:
                if v is not None and not np.isnan(float(v)):
                    vals.append(float(v))
            except (TypeError, ValueError):
                pass
        agg[k + "_mean"] = round(float(np.mean(vals)), 4) if vals else None
    agg["n_da_in_panel"]       = per_pert_list[0]["n_da_in_panel"]
    agg["n_test_perturbations"] = len(per_pert_list)
    return agg


# ── scGen encoder helper ───────────────────────────────────────────────────────
def encode_scgen(model, adata_subset):
    X = adata_subset.X
    if sp.issparse(X):
        X = X.toarray()
    x_tensor = torch.tensor(X, dtype=torch.float32).to(model.device)
    with torch.no_grad():
        _, _, z = model.module.z_encoder(x_tensor)
    return z.cpu().numpy()


# ── Evaluate scGen ────────────────────────────────────────────────────────────
def evaluate_scgen(cfg):
    import scgen
    label = cfg["label"]
    print(f"\n{'='*55}\n  scGen — {label}\n{'='*55}")

    if not os.path.exists(cfg["model_dir"]):
        print(f"  Model not found at {cfg['model_dir']} — skipping scGen {label}")
        return None

    adata = ad.read_h5ad(cfg["input"])
    gene_names = list(adata.var_names)
    gene_idx   = {g: i for i, g in enumerate(gene_names)}
    da_idx     = [gene_idx[g] for g in DA_MARKERS if g in gene_idx]
    print(f"  DA markers in panel: {len(da_idx)}/{len(DA_MARKERS)}")

    # Same split as 03_run_scgen.py (seed=42, EVAL_PERTS=20)
    np.random.seed(SEED)
    all_perts  = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
    counts     = adata.obs["condition"].value_counts()
    valid_perts = [p for p in all_perts if counts.get(p, 0) >= 20]
    eval_perts  = sorted(np.random.choice(valid_perts, size=EVAL_PERTS, replace=False))

    train_idx, test_idx = [], []
    ctrl_idx = adata.obs.index[adata.obs["condition"] == "ctrl"].tolist()
    train_idx.extend(ctrl_idx)
    for pert in eval_perts:
        idx = adata.obs.index[adata.obs["condition"] == pert].tolist()
        n_test = max(1, int(len(idx) * TEST_FRAC))
        np.random.shuffle(idx)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    for pert in [c for c in all_perts if c not in eval_perts]:
        idx = adata.obs.index[adata.obs["condition"] == pert].tolist()
        train_idx.extend(idx)

    adata_train = adata[train_idx].copy()
    adata_test  = adata[test_idx].copy()
    for a in [adata_train, adata_test]:
        a.obs["perturbation"] = a.obs["condition"].apply(
            lambda c: "ctrl" if c == "ctrl" else c.replace("+ctrl", ""))
        a.obs["cell_type"] = "iPSC-induced-neuron"

    # Load saved model
    print(f"  Loading scGen model from {cfg['model_dir']}...")
    scgen.SCGEN.setup_anndata(adata_train, batch_key="perturbation",
                               labels_key="cell_type")
    model = scgen.SCGEN.load(cfg["model_dir"], adata=adata_train)
    model.module.eval()

    ctrl_train    = adata_train[adata_train.obs["perturbation"] == "ctrl"].copy()
    z_ctrl_train  = encode_scgen(model, ctrl_train)
    z_ctrl_mean   = z_ctrl_train.mean(axis=0)
    X_ctrl        = (ctrl_train.X.toarray() if sp.issparse(ctrl_train.X)
                     else np.array(ctrl_train.X))
    ctrl_mean_expr = X_ctrl.mean(axis=0)

    per_pert_list = []
    for pert in eval_perts:
        pert_name   = pert.replace("+ctrl", "")
        true_cells  = adata_test[adata_test.obs["condition"] == pert]
        if true_cells.n_obs == 0:
            continue
        X_true     = (true_cells.X.toarray() if sp.issparse(true_cells.X)
                      else np.array(true_cells.X))
        truth_mean = X_true.mean(axis=0)

        pert_train = adata_train[adata_train.obs["perturbation"] == pert_name].copy()
        if pert_train.n_obs == 0:
            continue
        try:
            z_pert   = encode_scgen(model, pert_train).mean(axis=0)
            delta    = z_pert - z_ctrl_mean
            z_pred   = z_ctrl_train + delta
            z_tensor = torch.tensor(z_pred, dtype=torch.float32)
            with torch.no_grad():
                pred_expr = model.module.generative(z_tensor)["px"].cpu().numpy()
            pred_mean = pred_expr.mean(axis=0)
        except Exception as e:
            print(f"  Skipping {pert_name}: {e}")
            continue

        m = full_metrics_for_condition(pred_mean, truth_mean, ctrl_mean_expr,
                                        gene_names, DA_MARKERS)
        m["condition"] = pert_name
        per_pert_list.append(m)
        print(f"  {pert_name:25s} Pearson DE={m['pearson_de']:.3f}  "
              f"delta_DE={m['pearson_delta_de']:.3f}  "
              f"sign_DE={m['sign_accuracy_de']:.2f}")

    if not per_pert_list:
        print("  No valid perturbations evaluated.")
        return None

    agg = aggregate(per_pert_list)
    agg["model"]    = "scGen"
    agg["modality"] = label
    agg["eval_protocol"] = f"seen ({EVAL_PERTS} perts, 80/20 cell split)"
    with open(cfg["out_scgen"], "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Saved -> {cfg['out_scgen']}")
    return agg


# ── Evaluate mean baseline ────────────────────────────────────────────────────
def evaluate_baseline(cfg):
    label = cfg["label"]
    print(f"\n{'='*55}\n  Mean Baseline — {label}\n{'='*55}")

    adata = ad.read_h5ad(cfg["input"])
    gene_names = list(adata.var_names)

    np.random.seed(SEED)
    all_perts  = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
    np.random.shuffle(all_perts)
    test_perts  = set(all_perts[:EVAL_PERTS])
    train_perts = set(all_perts[EVAL_PERTS:])

    adata_train = adata[adata.obs["condition"].isin(train_perts)].copy()
    adata_test  = adata[adata.obs["condition"].isin(test_perts)].copy()
    adata_ctrl  = adata[adata.obs["condition"] == "ctrl"].copy()

    X_ctrl = (adata_ctrl.X.toarray() if sp.issparse(adata_ctrl.X)
               else np.array(adata_ctrl.X))
    ctrl_mean = X_ctrl.mean(axis=0)

    X_train = (adata_train.X.toarray() if sp.issparse(adata_train.X)
                else np.array(adata_train.X))
    global_pred = X_train.mean(axis=0)

    per_pert_list = []
    for pert in sorted(test_perts):
        true_cells = adata_test[adata_test.obs["condition"] == pert]
        if true_cells.n_obs == 0:
            continue
        X_true     = (true_cells.X.toarray() if sp.issparse(true_cells.X)
                      else np.array(true_cells.X))
        truth_mean = X_true.mean(axis=0)
        m = full_metrics_for_condition(global_pred, truth_mean, ctrl_mean,
                                        gene_names, DA_MARKERS)
        m["condition"] = pert
        per_pert_list.append(m)

    agg = aggregate(per_pert_list)
    agg["model"]    = "Mean Baseline"
    agg["modality"] = label
    agg["eval_protocol"] = f"seen ({EVAL_PERTS} perts)"
    with open(cfg["out_base"], "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved -> {cfg['out_base']}")
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []
    for cfg in DATASETS:
        base = evaluate_baseline(cfg)
        scgen_res = evaluate_scgen(cfg)
        if base:
            all_results.append(base)
        if scgen_res:
            all_results.append(scgen_res)

    print("\n\n" + "="*70)
    print("  SUMMARY TABLE (primary metrics)")
    print("="*70)
    print(f"  {'Model':<20} {'Mod':<8} {'delta_DE':>10} {'delta_DA':>10} "
          f"{'sign_DE':>10} {'sign_DA':>10} {'Pearson_DE':>12}")
    print(f"  {'-'*70}")
    for r in all_results:
        print(f"  {r.get('model',''):<20} {r.get('modality',''):<8} "
              f"{str(r.get('pearson_delta_de_mean','—')):>10} "
              f"{str(r.get('pearson_delta_da_mean','—')):>10} "
              f"{str(r.get('sign_accuracy_de_mean','—')):>10} "
              f"{str(r.get('sign_accuracy_da_mean','—')):>10} "
              f"{str(r.get('pearson_de_mean','—')):>12}")

    print("\nDone. Check results/scgen/ and results/baseline/ for updated JSONs.")
