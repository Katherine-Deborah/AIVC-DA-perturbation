"""
03b_run_scgen_crispr_a.py — scGen on CRISPRa (gene activation) dataset.
Same protocol as 03_run_scgen.py but on GSE152988 CRISPRa.
"""
import numpy as np, pandas as pd, json, os, anndata as ad
import scipy.sparse as sp, torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import scgen

INPUT      = "data/crispr_a_processed.h5ad"
RESULT_DIR = "results/scgen_crispra"
MODEL_DIR  = "results/scgen_crispra/model"
EPOCHS     = 100; BATCH_SIZE = 32; EVAL_PERTS = 20; TEST_FRAC = 0.2
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading CRISPRa data...")
adata = ad.read_h5ad(INPUT)
print(f"  Shape: {adata.shape}")

np.random.seed(42)
all_perts = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
np.random.shuffle(all_perts)
counts = adata.obs["condition"].value_counts()
valid  = [p for p in all_perts if counts.get(p, 0) >= 20]
eval_perts = sorted(np.random.choice(valid, size=min(EVAL_PERTS, len(valid)), replace=False))

train_idx, test_idx = [], []
train_idx.extend(adata.obs.index[adata.obs["condition"] == "ctrl"].tolist())
for pert in eval_perts:
    idx = adata.obs.index[adata.obs["condition"] == pert].tolist()
    n_test = max(1, int(len(idx) * TEST_FRAC))
    np.random.shuffle(idx); test_idx.extend(idx[:n_test]); train_idx.extend(idx[n_test:])
for pert in [c for c in all_perts if c not in eval_perts]:
    train_idx.extend(adata.obs.index[adata.obs["condition"] == pert].tolist())

adata_train = adata[train_idx].copy(); adata_test = adata[test_idx].copy()
print(f"  Train: {adata_train.n_obs:,} | Test: {adata_test.n_obs:,}")

for a in [adata_train, adata_test]:
    a.obs["perturbation"] = a.obs["condition"].apply(lambda c: "ctrl" if c == "ctrl" else c.replace("+ctrl",""))
    a.obs["cell_type"] = "iPSC-induced-neuron"

scgen.SCGEN.setup_anndata(adata_train, batch_key="perturbation", labels_key="cell_type")
model = scgen.SCGEN(adata_train)
print("Training scGen on CRISPRa...")
model.train(max_epochs=EPOCHS, batch_size=BATCH_SIZE, early_stopping=True, early_stopping_patience=10)
model.save(MODEL_DIR, overwrite=True)

def encode(model, a):
    X = a.X.toarray() if sp.issparse(a.X) else np.array(a.X)
    with torch.no_grad():
        _, _, z = model.module.z_encoder(torch.tensor(X, dtype=torch.float32).to(model.device))
    return z.cpu().numpy()

ctrl_train = adata_train[adata_train.obs["perturbation"] == "ctrl"].copy()
z_ctrl = encode(model, ctrl_train); z_ctrl_mean = z_ctrl.mean(axis=0)
ctrl_expr = (ctrl_train.X.toarray() if sp.issparse(ctrl_train.X) else np.array(ctrl_train.X)).mean(axis=0)

pa, pt, ms, r2s = [], [], [], []
print("\nEvaluating...")
for pert in eval_perts:
    pn = pert.replace("+ctrl","")
    tc = adata_test[adata_test.obs["condition"] == pert]
    if tc.n_obs == 0: continue
    tm = (tc.X.toarray() if sp.issparse(tc.X) else np.array(tc.X)).mean(axis=0)
    pt_train = adata_train[adata_train.obs["perturbation"] == pn].copy()
    if pt_train.n_obs == 0: continue
    try:
        delta = encode(model, pt_train).mean(axis=0) - z_ctrl_mean
        pred = np.array(model.module.generative(torch.tensor(z_ctrl + delta, dtype=torch.float32))["px"].cpu().detach()).mean(axis=0)
    except Exception as e: print(f"  Skip {pn}: {e}"); continue
    r,_ = pearsonr(pred, tm); pa.append(r)
    top20 = np.argsort(np.abs(tm - ctrl_expr))[-20:]
    r2,_ = pearsonr(pred[top20], tm[top20]); pt.append(r2)
    ms.append(mean_squared_error(tm, pred))
    ss_res=np.sum((tm-pred)**2); ss_tot=np.sum((tm-tm.mean())**2)
    r2s.append(1-ss_res/ss_tot if ss_tot>0 else 0.)
    print(f"  {pn}: Pearson={r:.3f}, top20={r2:.3f}")

metrics = {"pearson_all_mean": round(float(np.mean(pa)),4), "pearson_top20_mean": round(float(np.mean(pt)),4),
           "mse_mean": round(float(np.mean(ms)),4), "r2_mean": round(float(np.mean(r2s)),4),
           "n_test_perturbations": len(pa)}
with open(f"{RESULT_DIR}/scgen_metrics.json","w") as f: json.dump(metrics,f,indent=2)
print("\n=== scGen CRISPRa Results ===")
for k,v in metrics.items(): print(f"  {k}: {v}")
