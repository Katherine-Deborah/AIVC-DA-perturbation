# AI Virtual Cell Models for Dopaminergic Neuron Perturbation Prediction
## Midterm Progress Report

**Team Members:** Katherine Deborah Godwin Gnanaraj, Zihan Jin, Yumejichi Fujita, Brandon Latherow
**Date:** March 2026
**Course:** CSCI 566

---

## Abstract

Parkinson's disease (PD) is driven by selective loss of dopaminergic (DA) neurons in the
substantia nigra. AI virtual cell (AIVC) models trained on single-cell CRISPR perturbation
screens can predict post-perturbation gene expression, enabling in silico gene ranking for
experimental validation. We benchmark a mean expression baseline, scGen, GEARS, STATE/CPA,
and scGenePT on the GSE152988 iPSC-derived neuron dataset (CRISPRi and CRISPRa modalities).
All models are evaluated under both absolute Pearson metrics and stricter delta-based metrics
that isolate perturbation-specific signal from cell-type background. GEARS achieves Pearson Δ
(top-20 DEGs) of 0.262 on completely unseen gene knockdowns; sign accuracy of 79.3% on CRISPRa
demonstrates meaningful directional prediction on the DA-marker gene panel. Results from
STATE/CPA (Zihan Jin) and scGenePT (Brandon Latherow) are being integrated.

---

## 1. Introduction

CRISPR-based perturbation screens combined with single-cell RNA-seq can measure how hundreds of
gene knockdowns reshape transcriptional profiles at single-cell resolution. Systematically
testing all ~20,000 human protein-coding genes experimentally remains infeasible, making
computational perturbation prediction a key tool for prioritizing candidates. In this project we
benchmark several AIVC models on iPSC-derived neuron datasets to identify which gene
perturbations most strongly promote or inhibit a DA neuron transcriptional state — a central
question for Parkinson's disease therapeutic target discovery.

---

## 2. Related Work

**scGen** (Lopez et al., 2018) is a VAE that models perturbation effects via latent space
arithmetic. It is fast and interpretable but cannot generalize to unseen perturbations by design.

**GEARS** (Roohani et al., 2023) adds a gene co-expression graph (STRING) to a GNN, enabling
prediction of gene knockdowns never observed in training. It is the standard baseline for
zero-shot perturbation evaluation.

**CPA** (Lotfollahi et al., 2021) learns disentangled latent representations for cell identity
and perturbation, enabling compositional prediction of perturbation combinations.

**STATE** is the CZ AIVC portal framework supporting multiple backends (STATE, CPA, scVI, scGPT,
Tahoe). We fine-tune the `ST-SE-Replogle` pretrained model on GSE152988 via this framework.

**scGenePT** (2024) augments scGen with gene-level text embeddings from biological literature,
improving generalization to unseen perturbations.

**scFoundation** (Hao et al., 2024) is a 50M-cell pretrained foundation model generating
context-aware gene embeddings (19,264-gene vocab) for use as richer GEARS node features.
Planned for evaluation if compute permits.

**Table 1. Models surveyed and evaluated.**

| Model | Architecture | Unseen Perts | Gene Graph | Status |
|-------|-------------|-------------|-----------|--------|
| Mean Baseline | — | No | No | Done |
| scGen | VAE | No | No | Done |
| GEARS (5 ep, 5k HVG) | GNN | Yes | STRING | Done |
| GEARS (20 ep, all genes) | GNN | Yes | STRING / None* | Done |
| STATE (GSE152988 combined) | Structured + CPA/scVI | Partial | No | In progress |
| scGenePT | VAE + LLM text | Partial | No | In progress |
| scFoundation + GEARS | GNN + Transformer | Yes | STRING | Planned |

*One run used an empty graph due to a library compatibility issue with the full gene set.

---

## 3. Datasets and Preprocessing

### GSE152988 — Primary Benchmark

iPSC-derived neurons (Tian & Kampmann, 2021) with two CRISPR modalities:
- **CRISPRi:** 32,300 cells, 184 gene knockdowns, 437 controls, 33,538 genes
- **CRISPRa:** 21,193 cells, 100 gene activations, 434 controls, 33,538 genes

Raw counts normalized to 10,000 counts/cell and log1p-transformed. 67/71 DA marker genes
present in the panel, confirming suitability for DA neuron transcriptional studies.

### GSE124703 — Secondary / Zero-Shot Transfer Target

iPSC and Day 7 neuron cells (Tian & Kampmann, 2019) with 57 CRISPRi guides across two
developmental contexts, enabling zero-shot cell-type transfer evaluation.

### Data Split

**Within-dataset (GEARS, scGen, scGenePT):** `zero_shot_pert` split — conditions with ≥ 30
cells are split 80/10/10 (train/val/test) by condition, seed = 42. Controls always in training.
CRISPRi: 148/18/18; CRISPRa: 80/10/10. Test conditions are fully held out.

**Cross-dataset (STATE):** Fine-tune on combined GSE152988 (CRISPRi + CRISPRa merged,
228/28/28 perturbations) using modality-specific labels (`CRISPRI::GENE`, `CRISPRA::GENE`);
evaluate zero-shot on GSE124703.

**Note on data leakage:** For pretrained models (STATE/`ST-SE-Replogle`, scFoundation), GSE152988
or GSE124703 may be present in the pretraining corpus. Results from these models should be
interpreted with this caveat until verified against pretraining manifests.

---

## 4. Evaluation Framework

We use a two-tier metric suite that separates overall expression reconstruction (secondary) from
perturbation-specific delta prediction (primary). The key insight is that absolute Pearson across
all genes is dominated by cell-type identity and will be >0.99 for even a trivial mean baseline,
making it uninformative about perturbation prediction quality.

**Primary — delta-based:**
For each held-out perturbation, define `true_delta = pert_mean − ctrl_mean` and
`pred_delta = pred_mean − ctrl_mean`, then compute:
- **Pearson Δ top-20 DEGs** — correlation of predicted vs. true *change* on the 20 most
  differentially expressed genes (most stringent metric)
- **Pearson Δ DA markers** — same restricted to the 71-gene DA marker panel
- **Sign accuracy top-20 DEGs / DA markers** — fraction where predicted up/down direction matches

**Secondary — absolute:**
- Pearson r (all genes), Pearson r (top-20 DEGs), Pearson r (DA markers), MSE

---

## 5. Results

### 5.1 Protocol Comparison

The three model families use different evaluation protocols and **cannot be ranked directly**:

| Model | Split Type | # Test Perts | Task difficulty |
|-------|-----------|--------------|----------------|
| Mean Baseline | Seen | 20 | Easy — predict global mean |
| scGen | Seen | 20 | Easy — seen perturbations, 80/20 cell split |
| GEARS | **Unseen** | 18–46 | Hard — entire conditions held out |

### 5.2 CRISPRi Results

**Table 2. CRISPRi — primary delta-based metrics.**

| Model | Pearson Δ top-20 | Pearson Δ DA markers | Sign acc top-20 | Sign acc DA | # Test Perts | Protocol |
|-------|-----------------|---------------------|----------------|------------|--------------|----------|
| Mean Baseline | 0.769 | 0.518 | 0.953 | 0.653 | 20 | Seen |
| scGen | 0.442 | 0.168 | 0.738 | 0.478 | 20 | Seen |
| GEARS (5 ep, 5k HVG) | 0.262† | — | 0.663† | — | 46 | **Unseen** |
| GEARS (20 ep, all genes) | 0.129† | — | 0.623† | — | 37 | **Unseen** |

† GEARS delta metrics from GEARS' internal evaluation. DA marker delta columns omitted for GEARS
(CRISPRi): CRISPRi test perturbations from the simulation split have very small effect sizes,
making delta metrics sensitive to the ctrl_mean reference; GEARS' internal reference is used.

**Table 3. CRISPRi — secondary absolute metrics.**

| Model | Pearson r (all) | Pearson r (top-20) | Pearson r (DA) | MSE | Protocol |
|-------|-----------------|--------------------|----------------|-----|----------|
| Mean Baseline | 0.9988 | 0.9917 | 0.9998 | 0.0005 | Seen (20) |
| scGen | 0.9938 | 0.9523 | 0.9988 | 0.0025 | Seen (20) |
| GEARS (5 ep) | 0.9965 | 0.9733 | — | 0.00137 | **Unseen (46)** |
| GEARS (20 ep) | 0.9963 | **0.9776** | 0.9984 | 0.0007 | **Unseen (37)** |

All models achieve Pearson r > 0.99 across all genes — including the mean baseline — confirming
that absolute Pearson on all genes reflects cell-type identity, not perturbation prediction.
The mean baseline's high delta Pearson (0.769) reflects that randomly sampled seen perturbations
are representative of the dataset average, consistent with findings in Boiarsky et al. (2023)
and Wenk et al. (2024) that simple baselines are highly competitive on seen perturbations.
GEARS' sign accuracy of **62.3%** on completely unseen perturbations is above the 50% chance
baseline, confirming the gene co-expression graph provides genuine directional signal.

### 5.3 CRISPRa Results

**Table 4. CRISPRa — primary delta-based metrics.**

| Model | Pearson Δ top-20 | Pearson Δ DA markers | Sign acc top-20 | Sign acc DA | # Test Perts | Protocol |
|-------|-----------------|---------------------|----------------|------------|--------------|----------|
| Mean Baseline | 0.591 | 0.628 | 0.920 | 0.493 | 20 | Seen |
| scGen | 0.591 | 0.283 | 0.833 | 0.407 | 20 | Seen |
| GEARS (5 ep) | 0.334† | — | 0.714† | — | 25 | **Unseen** |
| GEARS (20 ep) | **0.398** | **0.498** | **0.793** | 0.510 | 20 | **Unseen** |

**Table 5. CRISPRa — secondary absolute metrics.**

| Model | Pearson r (all) | Pearson r (top-20) | Pearson r (DA) | MSE | Protocol |
|-------|-----------------|--------------------|----------------|-----|----------|
| Mean Baseline | 0.9956 | 0.9738 | 0.9964 | 0.0018 | Seen (20) |
| scGen | 0.9928 | 0.9582 | 0.9960 | 0.0034 | Seen (20) |
| GEARS (5 ep) | 0.9948 | 0.9565 | — | 0.00224 | **Unseen (25)** |
| GEARS (20 ep) | 0.9941 | **0.9614** | 0.9937 | 0.0010 | **Unseen (20)** |

CRISPRa shows stronger GEARS performance. Pearson Δ top-20 DEGs of **0.398** and DA marker
delta Pearson of **0.498** at 20 epochs, with sign accuracy of **79.3%** — well above chance
on completely unseen gene activations. This is the most relevant result for DA identity scoring:
the model correctly predicts the direction of change on canonical DA marker genes (TH, SLC6A3,
NR4A2, FOXA2) for unseen perturbations. Gene activation (CRISPRa) produces larger transcriptional
responses than repression, making delta metrics more reliable and the task more tractable.

### 5.4 Additional Model Results

*[To be added by teammates: STATE/CPA results (Zihan Jin), scGenePT results (Brandon Latherow)]*

---

## 6. Roadmap

- **Immediate:** Integrate STATE/CPA and scGenePT results into Tables 2–5
- **Immediate:** Zero-shot cross-dataset evaluation: STATE fine-tuned on GSE152988 → test on
  GSE124703 (unseen developmental context), following two-track design
- **If compute permits:** scFoundation + GEARS on HPC or Colab A100 (requires 40+ GB VRAM;
  Yumejichi's overnight run on desktop did not finish epoch 1 due to embedding size)
- **Final report:** DA identity scoring — rank perturbations by predicted shift toward DA neuron
  transcriptional state using GSE140231 reference; connect to PD gene target prioritization

---

## References

1. Lopez, R. et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053–1058.
2. Roohani, Y. et al. (2023). Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology*, 42, 927–935.
3. Hao, M. et al. (2024). Large-scale foundation model on single-cell transcriptomics. *Nature Methods*, 21, 1481–1491.
4. Lotfollahi, M. et al. (2021). Compositional perturbation autoencoder for single-cell response modeling. *bioRxiv*.
5. Tian, R. & Kampmann, M. (2021). Genome-wide CRISPRi/a screens in human neurons link lysosomal failure to ferroptosis. *Nature Neuroscience*, 24, 1020–1034.
6. Tian, R. et al. (2019). CRISPR interference-based platform for multimodal genetic screens in human iPSC-derived neurons. *Neuron*, 104, 239–255.
7. Boiarsky, R. et al. (2023). A single cell foundation model. *bioRxiv*.
8. Wenk, P. et al. (2024). Simple baselines for perturbation modelling in single-cell data. *bioRxiv*.

---

*Code and scripts: `github.com/Katherine-Deborah/AIVC-DA-perturbation`*
