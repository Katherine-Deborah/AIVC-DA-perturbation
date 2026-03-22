# 2026-03-21


## Split Style Used

`GSE152988` is currently prepared as three `STATE`-ready datasets:

- `crispri`
- `crispra`
- `combined`

The current training plan uses the `combined` dataset as the main fine-tuning entry. In that
combined dataset:

- `CRISPRi` and `CRISPRa` cells are merged into one training set
- perturbation labels are modality-specific for non-controls, e.g. `CRISPRI::GENE` and
  `CRISPRA::GENE`
- control cells remain labeled as `control`
- batch labels are modality-prefixed to avoid collisions across the two source datasets

For each prepared dataset, the current repo uses only `zero_shot_pert` style splitting:

- controls stay in training
- perturbations with at least `30` cells are eligible
- eligible perturbations are split into train / val / test with:
  - `VAL_FRAC = 0.1`
  - `TEST_FRAC = 0.1`
- the split is deterministic with `SEED = 42`

Observed split summaries after preprocessing:

- `crispri`: `148` train perturbations, `18` val perturbations, `18` test perturbations
- `crispra`: `80` train perturbations, `10` val perturbations, `10` test perturbations
- `combined`: `228` train perturbations, `28` val perturbations, `28` test perturbations; this is
  the main fine-tuning dataset for the current benchmark plan

