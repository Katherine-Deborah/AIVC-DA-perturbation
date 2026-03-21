"""
02b_run_gears_crispr_a.py
Train and evaluate GEARS on GSE152988 CRISPRa data.
Outputs: results/gears_crispra/gears_metrics.json
"""
import numpy as np
import json
import os
import anndata as ad

from gears import PertData, GEARS
from gears.gears import evaluate, compute_metrics

INPUT       = "data/crispr_a_processed.h5ad"
DATA_DIR    = "data/gears_data"
RESULT_DIR  = "results/gears_crispra"
DATASET     = "crispra_da"
EPOCHS      = 5
BATCH_SIZE  = 32
DEVICE      = "cuda"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading processed data...")
adata = ad.read_h5ad(INPUT)
print(f"  Shape: {adata.shape}")

# ── 2. Build PertData object ───────────────────────────────────────────────────
print("\nBuilding GEARS PertData...")
pert_data = PertData(DATA_DIR)

processed_path = os.path.join(DATA_DIR, DATASET, "perturb_processed.h5ad")
if os.path.exists(processed_path):
    print("  Found cached processed data — loading...")
    pert_data.load(data_path=os.path.join(DATA_DIR, DATASET))
else:
    print("  Processing new dataset...")
    pert_data.new_data_process(dataset_name=DATASET, adata=adata, skip_calc_de=False)

# ── 3. Split ───────────────────────────────────────────────────────────────────
print("\nPreparing splits...")
pert_data.prepare_split(split="simulation", seed=42)
pert_data.get_dataloader(batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)

# ── 4. Train ───────────────────────────────────────────────────────────────────
print("\nTraining GEARS...")
gears_model = GEARS(pert_data, device=DEVICE)
gears_model.model_initialize(hidden_size=64)
gears_model.train(epochs=EPOCHS)

# ── 5. Evaluate using GEARS internal functions ────────────────────────────────
print("\nRunning final test evaluation...")
test_loader = pert_data.dataloader["test_loader"]
test_res = evaluate(test_loader, gears_model.best_model,
                    gears_model.pert_list, gears_model.device)
test_metrics, test_pert_res = compute_metrics(test_res)

metrics = {
    "pearson_all_mean":   round(float(test_metrics.get("pearson", 0)), 4),
    "pearson_top20_mean": round(float(test_metrics.get("pearson_de", 0)), 4),
    "mse_mean":           round(float(test_metrics.get("mse", 0)), 4),
    "mse_top20_mean":     round(float(test_metrics.get("mse_de", 0)), 4),
    "n_test_perturbations": len(test_pert_res),
    "split": "unseen_single (simulation)",
}

with open(os.path.join(RESULT_DIR, "gears_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== GEARS CRISPRa Results ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print(f"\nSaved to {RESULT_DIR}/gears_metrics.json")
