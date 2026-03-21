# GEARS 5-Epoch Results (Preliminary)

Saved before re-running with 15 epochs.

## CRISPRi (Gene Knockdown) — 5 epochs

| Metric | Value |
|--------|-------|
| Pearson r (all genes) | 0.997 |
| Pearson r (top-20 DEGs) | 0.973 |
| MSE | 0.00137 |
| Pearson delta | 0.262 |
| Frac wrong direction (top-20) | 33.7% |
| Unseen perturbations tested | 46 |
| Split | simulation (unseen_single) |

### Validation curve (Val Top-20 DE MSE)
| Epoch | Val Overall MSE | Val Top-20 DE MSE |
|-------|----------------|-------------------|
| 1 | 0.0016 | 0.0044 |
| 2 | 0.0016 | 0.0044 |
| 3 | 0.0018 | 0.0044 |
| 4 | 0.0014 | 0.0041 |
| 5 | 0.0013 | 0.0041 |

*Still improving at epoch 5 — not fully converged.*

---

## CRISPRa (Gene Activation) — 5 epochs

| Metric | Value |
|--------|-------|
| Pearson r (all genes) | 0.995 |
| Pearson r (top-20 DEGs) | 0.957 |
| MSE | 0.00224 |
| Pearson delta | 0.334 |
| Frac wrong direction (top-20) | 28.6% |
| Unseen perturbations tested | 25 |
| Split | simulation (unseen_single) |

### Validation curve (Val Top-20 DE MSE)
| Epoch | Val Overall MSE | Val Top-20 DE MSE |
|-------|----------------|-------------------|
| 1 | 0.0019 | 0.0112 |
| 2 | 0.0018 | 0.0100 |
| 3 | 0.0036 | 0.0121 |
| 4 | 0.0015 | 0.0096 |
| 5 | 0.0019 | 0.0094 |

*Still clearly improving at epoch 5 — not fully converged.*
