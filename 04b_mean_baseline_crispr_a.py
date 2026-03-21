"""Mean baseline on CRISPRa data."""
import numpy as np, json, os, anndata as ad
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

INPUT = "data/crispr_a_processed.h5ad"; RESULT_DIR = "results/baseline_crispra"; N_TEST = 20
os.makedirs(RESULT_DIR, exist_ok=True)
adata = ad.read_h5ad(INPUT)
np.random.seed(42)
all_perts = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
np.random.shuffle(all_perts)
test_perts = set(all_perts[:N_TEST]); train_perts = set(all_perts[N_TEST:])
adata_train = adata[adata.obs["condition"].isin(train_perts)].copy()
adata_test  = adata[adata.obs["condition"].isin(test_perts)].copy()
X_train = adata_train.X.toarray() if sp.issparse(adata_train.X) else np.array(adata_train.X)
global_mean = X_train.mean(axis=0)
pa, pt, ms, r2s = [], [], [], []
for pert in sorted(test_perts):
    tc = adata_test[adata_test.obs["condition"] == pert]
    if tc.n_obs == 0: continue
    tm = (tc.X.toarray() if sp.issparse(tc.X) else np.array(tc.X)).mean(axis=0)
    r,_ = pearsonr(global_mean, tm); pa.append(r)
    r2,_ = pearsonr(global_mean[np.argsort(tm)[-20:]], tm[np.argsort(tm)[-20:]]); pt.append(r2)
    ms.append(mean_squared_error(tm, global_mean))
    ss_res=np.sum((tm-global_mean)**2); ss_tot=np.sum((tm-tm.mean())**2)
    r2s.append(1-ss_res/ss_tot if ss_tot>0 else 0.)
metrics = {"pearson_all_mean": round(float(np.mean(pa)),4), "pearson_top20_mean": round(float(np.mean(pt)),4),
           "mse_mean": round(float(np.mean(ms)),4), "r2_mean": round(float(np.mean(r2s)),4), "n_test_perturbations": len(pa)}
with open(f"{RESULT_DIR}/baseline_metrics.json","w") as f: json.dump(metrics,f,indent=2)
print("=== CRISPRa Mean Baseline ===")
for k,v in metrics.items(): print(f"  {k}: {v}")
