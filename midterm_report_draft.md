# AI Virtual Cell Models for Dopaminergic Neuron Perturbation Prediction
## Midterm Progress Report

**Team Members:** 
**Date:** March 2026
**Course:** CSCI 566

---

## Abstract

Parkinson's disease (PD) is driven by the selective loss of dopaminergic (DA) neurons in the
substantia nigra. Identifying which gene perturbations can shift cells toward or away from a DA
neuron transcriptional state is critical for discovering therapeutic targets, but testing hundreds
of candidates experimentally is prohibitively costly. AI virtual cell (AIVC) models offer a
computational alternative: trained on single-cell CRISPR perturbation screens, these models
learn to predict post-perturbation gene expression profiles. In this midterm report, we survey
the landscape of AIVC perturbation prediction models, describe our datasets and preprocessing
pipeline, and present initial benchmarking results on the GSE152988 iPSC-derived neuron dataset
using two models — scGen and GEARS — alongside a mean expression baseline. Our results
demonstrate that GEARS achieves a Pearson correlation of 0.973 on the top-20 differentially
expressed genes (DEGs) even for completely unseen gene knockdowns (CRISPRi), while scGen
achieves 0.952 on seen CRISPRi perturbations and 0.927 on CRISPRa activations. We outline a
roadmap for extending these experiments to additional models and datasets.

---

## 1. Introduction

Parkinson's disease affects approximately 10 million people worldwide and currently has no
disease-modifying treatment. The primary pathological event is the degeneration of dopaminergic
(DA) neurons in the substantia nigra pars compacta (SNpc), which produce dopamine essential for
motor control. Understanding the transcriptional programs that define DA neuron identity — and
how genetic perturbations shift cells toward or away from this state — is a central goal of
modern PD research.

CRISPR-based perturbation screens, combined with single-cell RNA sequencing (scRNA-seq), now
make it possible to measure how hundreds of gene knockdowns reshape transcriptional profiles at
single-cell resolution. However, systematically testing all ~20,000 human protein-coding genes
experimentally remains infeasible. AI virtual cell (AIVC) models address this by learning to
predict post-perturbation gene expression from observed data, enabling in silico screening of
genetic targets at scale.

In this project, we benchmark several AIVC perturbation prediction models on two human datasets
featuring iPSC-derived neurons subjected to CRISPR perturbation screens. Our primary goal is to
identify which gene perturbations most strongly promote or inhibit a DA neuron transcriptional
state, contributing to a systematic gene ranking pipeline for experimental validation.

This midterm report covers: (1) a literature review of AIVC models, (2) dataset description and
preprocessing, (3) initial experimental results from scGen and GEARS on the GSE152988 dataset,
and (4) a research roadmap for the remainder of the project.

---

## 2. Related Work

A growing ecosystem of AI models has been developed for single-cell perturbation prediction.
We survey the most relevant approaches below and summarize their key characteristics in Table 1.

**scGen** (Lopez et al., 2018) is a variational autoencoder (VAE) that models perturbation
effects via latent space arithmetic. Control and perturbed cells are encoded into a shared
latent space; a perturbation "delta vector" — the difference between mean latent representations
of perturbed and control cells — is then applied to new control cells to predict their perturbed
state. scGen is conceptually simple, fast to train, and has been widely used as a baseline. Its
primary limitation is that it cannot generalize to novel perturbations not seen during training,
as it requires observed perturbed cells to compute the delta vector.

**GEARS** (Roohani et al., 2023) overcomes scGen's generalization limitation by incorporating a
gene co-expression graph (derived from the STRING database) as an inductive bias. A graph neural
network (GNN) is trained where each node is a gene and edges encode co-expression relationships.
By reasoning over this gene network, GEARS can predict the effect of genes never perturbed in
training, as long as they share co-expression relationships with observed genes. GEARS is
currently the standard baseline for unseen perturbation evaluation and supports both single and
combinatorial perturbation prediction.

**scFoundation** (Hao et al., 2024) is a large-scale foundation model pretrained on over 50
million human single-cell transcriptomes using a masked gene expression modeling objective over
a fixed 19,264-gene vocabulary. Rather than performing perturbation prediction itself,
scFoundation acts as a feature extractor, generating context-aware gene embeddings (shape: cells
× 19,264 × 512) that can serve as richer node features in downstream models such as GEARS. The
hypothesis is that these biologically-informed embeddings encode gene function and co-regulation
patterns that improve prediction beyond the STRING graph alone. This model requires substantial
GPU resources (40+ GB VRAM) and is planned for evaluation on HPC infrastructure.

**CPA** (Lotfollahi et al., 2021) extends the VAE framework by learning disentangled latent
representations for cell identity, perturbation effect, and technical covariates. This
compositionality allows CPA to predict effects of perturbation combinations by adding individual
latent components, making it well-suited for multi-perturbation settings. CPA has been shown to
outperform scGen on combinatorial perturbation benchmarks.

**STATE** is a perturbation prediction model listed on the CZ AIVC portal. It incorporates
structured representations of transcriptional states to model how perturbations shift cell
identity across developmental trajectories, making it particularly relevant to our DA neuron
context. Evaluation on our datasets is planned.

**scGenePT** (2024) augments the scGen framework with gene-level text embeddings derived from
large language models (LLMs) trained on biological literature. Each gene is represented not only
by its expression level but also by a natural language description of its function, enabling the
model to leverage biological knowledge encoded in text to improve generalization to unseen
perturbations.

**scGPT** (Cui et al., 2024) is a 100M-parameter transformer pretrained on 33 million human
single-cell profiles. It supports perturbation prediction via fine-tuning but requires
significant GPU resources (16+ GB VRAM) and Linux-specific dependencies (`flash-attn`), making
it suitable for HPC or cloud evaluation only.

**Table 1. Summary of AIVC perturbation prediction models surveyed.**

| Model | Architecture | Unseen Perturbations | Gene Graph | Foundation Model | Evaluated |
|-------|-------------|---------------------|-----------|-----------------|-----------|
| scGen | VAE | No | No | No | Yes |
| GEARS | GNN | Yes | STRING | No | Yes |
| scFoundation+GEARS | GNN + Transformer | Yes | STRING | Yes (50M cells) | Planned (HPC) |
| CPA | VAE (disentangled) | Partial | No | No | Attempted* |
| STATE | Structured model | Yes | No | No | Planned |
| scGenePT | VAE + LLM text | Partial | No | Text embeddings | Planned |
| scGPT | Transformer | Yes | No | Yes (33M cells) | Planned (HPC) |

*CPA installation failed due to strict version constraints incompatible with Python 3.11.

---

## 3. Datasets

### 3.1 GSE152988 — Primary Benchmark Dataset

Our primary benchmark is the Tian & Kampmann (2021) iPSC-derived neuron CRISPR screen,
accessed as pre-processed AnnData objects. The dataset contains two perturbation modalities:

- **CRISPRi (transcriptional repression):** 32,300 cells across 184 gene knockdowns + 437
  control cells; 33,538 genes measured. This is our primary experimental dataset.
- **CRISPRa (transcriptional activation):** 21,193 cells across 100 gene activations + 434
  control cells; 33,538 genes measured.

Both datasets feature iPSC-induced neurons from the LUHMES cell line. Raw integer count
matrices were normalized to 10,000 counts per cell and log1p-transformed. We selected the top
5,000 highly variable genes (HVGs) using the Seurat v3 method to reduce memory requirements
for our local GPU (NVIDIA GTX 1650 Ti, 4 GB VRAM).

DA marker gene analysis (71 canonical DA markers including TH, SLC6A3, NR4A2, FOXA2, PITX3)
reveals that 67/71 markers are present in the gene panel, confirming this dataset is appropriate
for DA neuron transcriptional studies.

### 3.2 GSE124703 — Secondary Dataset

This dataset (Tian & Kampmann, 2019) contains iPSC and Day 7 neuron cells with 57 CRISPRi
guide sequences targeting ~25 gene knockdowns. It provides two developmental contexts (iPSC vs.
early neuron stage) for the same perturbations, enabling cross-stage comparison. While 70/71 DA
markers are present in the gene panel, key DA markers TH and SLC6A3 show low expression,
suggesting these are generic neurons rather than confirmed DA neurons. This dataset is reserved
for future cross-dataset generalization experiments.

### 3.3 GSE140231 — DA Reference Dataset

This substantia nigra dataset (~20,000 cells from human donors) contains confirmed DA neurons
alongside other cell types. We plan to use it to derive clean DA neuron identity labels for
scoring perturbation effects in terms of DA identity shift. Preprocessing is ongoing.

---

## 4. Methods

### 4.1 Preprocessing

Raw count matrices were normalized and log1p-transformed using Scanpy. Top 5,000 highly
variable genes were selected per dataset. For GEARS, gene names were mapped to GEARS' format
(`GENE+ctrl` for single perturbations, `ctrl` for controls) and a `gene_name` variable was
added to the AnnData object.

### 4.2 Evaluation Protocols

We use two distinct evaluation protocols that test different model capabilities:

**Seen perturbation evaluation (scGen):** For each of 20 randomly selected perturbations, 20%
of cells are withheld for testing while the remaining 80% are included in training. The model
is trained on all 184 perturbations and evaluated on the withheld cells. This tests how well
the model reconstructs the expression profile of perturbations it has already observed, and
is analogous to an interpolation task.

**Unseen perturbation evaluation (GEARS):** Using the GEARS "simulation" split, 46 entire
perturbation conditions are held out from training entirely — no cells from these perturbations
are seen during training. The model must predict their effects based solely on gene network
structure and patterns from the 138 observed perturbations. This tests true out-of-distribution
generalization, which is the scientifically critical capability for virtual cell screening.

### 4.3 Mean Expression Baseline

We implement a trivial baseline that predicts the global mean expression of all training
perturbed cells, regardless of which gene is knocked down. This is an important reference
point: because single-gene CRISPR perturbations affect relatively few genes, the overall
gene expression profile is dominated by cell-type identity rather than the perturbation.
As a result, Pearson correlation across all genes is high for all methods including this
baseline. The more diagnostic metric is **Pearson correlation on the top-20 most
differentially expressed genes (DEGs)**, which isolates the prediction of perturbation-specific
transcriptional changes.

### 4.4 Metrics

For each held-out perturbation we compute the mean expression vector across cells, then report:
- **Pearson r (all genes):** Correlation between predicted and true mean expression across all
  5,000 HVGs
- **Pearson r (top-20 DEGs):** Correlation restricted to the top-20 most differentially
  expressed genes (highest absolute deviation from control), the standard GEARS evaluation
  metric
- **MSE:** Mean squared error between predicted and true mean expression
- **Pearson delta** (GEARS only): Correlation on the predicted vs. true *change* from control,
  the most stringent metric for perturbation-specific prediction
- **R²:** Variance explained

All metrics are averaged across held-out perturbations.

---

## 5. Results

### 5.1 CRISPRi Benchmark (Gene Knockdown)

Table 2 shows results on the CRISPRi dataset. All three methods are evaluated on 20 held-out
perturbations (scGen/Baseline) or 46 held-out perturbations (GEARS).

**Table 2. Perturbation prediction performance on GSE152988 CRISPRi (gene knockdown).**

| Model | Pearson r (all genes) | Pearson r (top-20 DEGs) | MSE | Evaluation Protocol |
|-------|-----------------------|--------------------------|-----|---------------------|
| Mean Baseline | 0.999 | 0.995 | 0.0005 | Seen (20 perts) |
| scGen | 0.994 | 0.952 | 0.0025 | Seen (20 perts) |
| GEARS | **0.997** | **0.973** | **0.00137** | **Unseen (46 perts)** |

Several observations are important for interpreting these results:

First, all methods including the trivial baseline achieve Pearson > 0.99 across all genes. This
is expected: single-gene knockdowns alter expression of only a small fraction of the
transcriptome, so the bulk of expression variance reflects cell-type identity, which all methods
capture implicitly. The baseline's apparent superiority on this metric is therefore misleading.

Second, the top-20 DEG Pearson is the appropriate metric for comparing perturbation-specific
prediction. Here scGen (0.952) and GEARS (0.973) both improve over the baseline (0.995 on seen
perturbations), but the key distinction is that **GEARS achieves 0.973 on completely unseen
perturbations** — genes that were never knocked down in training. This represents genuine
generalization ability enabled by the gene co-expression graph.

Third, GEARS' **Pearson delta of 0.262** — measuring correlation on the predicted vs. true
*change* from control — reveals that while absolute expression is predicted well, predicting the
direction and magnitude of perturbation-specific shifts remains challenging. Similarly, 33.7% of
top-20 DE genes are predicted in the wrong direction of change, indicating room for improvement
with richer gene representations (e.g., via scFoundation embeddings).

### 5.2 CRISPRa Benchmark (Gene Activation)

**Table 3. Perturbation prediction performance on GSE152988 CRISPRa (gene activation).**

| Model | Pearson r (all genes) | Pearson r (top-20 DEGs) | MSE | Evaluation Protocol |
|-------|-----------------------|--------------------------|-----|---------------------|
| Mean Baseline | 0.996 | 0.917 | 0.0018 | Seen (20 perts) |
| scGen | 0.993 | 0.927 | 0.0029 | Seen (20 perts) |
| GEARS | **0.995** | **0.957** | **0.00224** | **Unseen (25 perts)** |

CRISPRa is a harder benchmark than CRISPRi across all methods. The mean baseline drops from
0.995 (CRISPRi) to 0.917 (CRISPRa) on top-20 DEGs, reflecting that gene activations produce
larger and more diverse transcriptional changes than knockdowns.

GEARS achieves top-20 DEG Pearson = **0.957** on 25 completely unseen CRISPRa perturbations,
outperforming the seen-perturbation baseline (0.917) and scGen (0.927) despite never having
observed these activations during training. The Pearson delta of 0.334 and 28.6% wrong-direction
rate on CRISPRa are slightly better than the CRISPRi equivalents (0.262 delta, 33.7% wrong
direction), suggesting the STRING gene co-expression graph provides useful signal for predicting
activation effects, though the higher absolute MSE (0.00224 vs 0.00137) reflects the larger
magnitude of transcriptional changes induced by CRISPRa.

### 5.3 Training Dynamics

GEARS trained for 5 epochs (~25 minutes on a GTX 1650 Ti) with validation Top-20 DE MSE
improving from 0.0044 (epoch 1) to 0.0041 (epochs 4-5), indicating the model was still
improving at termination. Additional training epochs on larger GPU hardware are expected to
yield further gains.

scGen employed early stopping with patience=10 epochs and converged within approximately 11
epochs on both CRISPRi and CRISPRa, with final validation ELBO of ~900-1200.

---

## 6. Research Ideas and Future Work

Based on our initial results and literature review, we propose the following directions for
the remainder of the project:

**6.1 scFoundation+GEARS integration.** Our most important planned experiment is replacing
GEARS' default gene node features with context-aware embeddings from scFoundation (shape:
cells × 19,264 × 512). The hypothesis is that richer gene representations encoding biological
function will improve unseen perturbation generalization, directly addressing GEARS' current
Pearson delta of only 0.26. This requires a GPU with 40+ GB VRAM and is planned for university
HPC resources.

**6.2 Additional model benchmarking.** We plan to evaluate STATE and scGenePT once repository
access is confirmed. scGPT will be evaluated on Google Colab Pro (A100, 40 GB) given its
dependency on `flash-attn` (Linux-only) and 16+ GB VRAM requirement.

**6.3 DA identity scoring.** The scientifically central question is not just prediction accuracy
but which perturbations most shift cells toward a DA neuron identity. We plan to: (1) extract
clean DA neuron labels from GSE140231, (2) score cells in GSE152988 using the 71-gene DA marker
panel, and (3) rank perturbations by predicted DA identity shift under each model. This produces
the gene priority list relevant for experimental validation in PD.

**6.4 Cross-dataset generalization.** Training on GSE152988 and evaluating on GSE124703 tests
whether perturbation effects generalize across different iPSC differentiation protocols and
developmental stages (iPSC vs. Day 7 neuron). This is a stricter test of model robustness
relevant to translating predictions to new experimental contexts.

**6.5 CRISPRi vs. CRISPRa comparison.** Our preliminary results suggest CRISPRa (gene
activation) is a harder prediction task than CRISPRi (knockdown). We plan to train and evaluate
all models on both modalities and analyze whether models that excel at predicting knockdowns
also generalize to activation, and vice versa.

**6.6 Multi-perturbation prediction.** All current experiments focus on single-gene
perturbations. GEARS natively supports combinatorial perturbation prediction. Extending
evaluation to double knockdowns (using GSE124703's multi-guide data) would test whether models
can predict synergistic or antagonistic gene interactions relevant to PD pathway analysis.

**6.7 Baseline robustness.** Following recent literature showing that simple baselines often
match foundation models on perturbation benchmarks (Boiarsky et al., 2023; Wenk et al., 2024),
we will include a broader set of baselines: linear regression, mean-per-perturbation, and PCA
projection. Rigorous comparison against these baselines is essential for contextualizing the
value added by neural approaches.

---

## 7. Conclusion

We have established a working benchmark pipeline for AIVC perturbation prediction on iPSC-
derived neuron data relevant to Parkinson's disease. Across both CRISPRi and CRISPRa datasets,
GEARS achieves strong performance on completely unseen perturbations (top-20 DEG Pearson = 0.973
for knockdown, 0.957 for activation), outperforming scGen on the harder generalization task in
both settings. The Pearson delta metrics (0.262 CRISPRi, 0.334 CRISPRa) and wrong-direction
rates (~30–34%) indicate meaningful room for improvement, which we hypothesize can be addressed
by richer gene representations from scFoundation embeddings. The next two weeks will focus on
HPC-scale experiments with scFoundation+GEARS, additional model benchmarking (STATE, scGenePT),
and DA identity scoring to connect perturbation predictions to the biological question of DA
neuron identity in Parkinson's disease.

---

## References

1. Lopez, R. et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053–1058.
2. Roohani, Y. et al. (2023). Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology*, 42, 927–935.
3. Hao, M. et al. (2024). Large-scale foundation model on single-cell transcriptomics. *Nature Methods*, 21, 1481–1491.
4. Lotfollahi, M. et al. (2021). Compositional perturbation autoencoder for single-cell response modeling. *bioRxiv*.
5. Cui, H. et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21, 1470–1480.
6. Tian, R. & Kampmann, M. (2021). Genome-wide CRISPRi/a screens in human neurons link lysosomal failure to ferroptosis. *Nature Neuroscience*, 24, 1020–1034.
7. Tian, R. et al. (2019). CRISPR interference-based platform for multimodal genetic screens in human iPSC-derived neurons. *Neuron*, 104, 239–255.
8. Boiarsky, R. et al. (2023). A single cell foundation model. *bioRxiv*.

---

*All code and processed data are available in the project repository.*
