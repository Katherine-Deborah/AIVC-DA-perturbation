"""
05_collect_results.py
Collect metrics from all models and print a comparison table.
Also generates a simple matplotlib comparison bar chart.
"""
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

results_files = {
    "Mean Baseline":   "results/baseline/baseline_metrics.json",
    "scGen":           "results/scgen/scgen_metrics.json",
    "GEARS":           "results/gears/gears_metrics.json",
}

metrics_to_show = ["pearson_all_mean", "pearson_top20_mean", "mse_mean", "r2_mean"]

all_results = {}
for model, path in results_files.items():
    if os.path.exists(path):
        with open(path) as f:
            all_results[model] = json.load(f)
    else:
        print(f"  [not found] {path}")

print("\n{'Model':<20} {'Pearson All':>12} {'Pearson Top20':>14} {'MSE':>10} {'R²':>8}")
print("-" * 70)
for model, metrics in all_results.items():
    pa   = metrics.get("pearson_all_mean", "N/A")
    pt20 = metrics.get("pearson_top20_mean", "N/A")
    mse  = metrics.get("mse_mean", "N/A")
    r2   = metrics.get("r2_mean", "N/A")
    print(f"{model:<20} {str(pa):>12} {str(pt20):>14} {str(mse):>10} {str(r2):>8}")

# Save combined results
combined = {"models": all_results, "metrics_key": metrics_to_show}
with open("results/combined_metrics.json", "w") as f:
    json.dump(combined, f, indent=2)
print("\nSaved: results/combined_metrics.json")

# Bar chart
if HAS_PLOT and len(all_results) > 0:
    models = list(all_results.keys())
    pearson_all   = [all_results[m].get("pearson_all_mean", 0) for m in models]
    pearson_top20 = [all_results[m].get("pearson_top20_mean", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, pearson_all,   width, label="Pearson (all genes)",   color="#4C72B0")
    ax.bar(x + width/2, pearson_top20, width, label="Pearson (top-20 DEGs)", color="#DD8452")

    ax.set_xlabel("Model")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Perturbation Prediction Performance\n(GSE152988 CRISPRi, held-out perturbations)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150)
    print("Saved: results/model_comparison.png")
