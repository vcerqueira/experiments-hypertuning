# Are Transformers Really Bad for Time Series Forecasting? A Hyperparameter Optimization Perspective

Code for the experiments in the paper *"Are Transformers Really Bad for Time Series Forecasting? A Hyperparameter Optimization Perspective"*.

This repository collects a large number of runs over neural forecasting architectures, then simulates and studies hyperparameter optimization from several perspectives: learning curves, overtuning, hyperparameter importance, and metalearning-based search.

## Setup

Set up your `DATA_DIR` in a `.env` file based on `.env.example`.
Run analysis scripts from the **repository root** so relative paths such as `assets/outputs` resolve correctly. For the experiment runner, either run from `scripts/experiments/` or adjust `results_dir` in that script.

### Dependencies

Required packages are listed in `requirements.txt` (notably `neuralforecast`, `datasets`, `modelradar`, `optuna`).

Datasets are loaded on demand:

- Monash / Chronos-style series via Hugging Face (`autogluon/chronos_datasets`)
- Long-horizon series (`TrafficL`, `ECL`) via `datasetsforecast`

### Hardware

Training uses PyTorch through NeuralForecast. On Apple Silicon, `TRY_MPS` in `src/config.py` enables MPS when available. Full random search is compute-heavy (hundreds of configs × models × datasets).

## Repository layout

```
scripts/
  experiments/experiments-run-all.py   # extended random search (CV results)
  analysis/
    2_learning_curve.py                # learning curves vs # sampled configs
    3_overtuning.py                    # overtuning trajectories
    4_hyperparams_importance.py        # fANOVA hyperparameter importance
    5_at.py                            # metalearning / active-testing search
    5_at_analysis.py                   # summarize AT vs RS results
src/                                   # loaders, model configs, utilities
assets/
  outputs/                             # figures from analyses 2–4
  results/                             # CSV outputs from AT search (step 5)
```

## Reproducing the experiments

Pipeline overview:

1. **Run extended random search** → inner/outer CV CSVs per model, dataset, and config  
2. **Learning curves** (`2_learning_curve.py`)  
3. **Overtuning** (`3_overtuning.py`)  
4. **Hyperparameter importance** (`4_hyperparams_importance.py`)  
5. **Metalearning-based search** (`5_at.py` → `5_at_analysis.py`)

### 0. Extended random search

Entry point: [`scripts/experiments/experiments-run-all.py`](scripts/experiments/experiments-run-all.py).

For each model and sampled hyperparameter configuration, the script:

1. Fits with NeuralForecast cross-validation on an **inner** (estimation) split  
2. Evaluates on an **outer** (full) split  
3. Writes CSV files named `{model},{dataset},{config_id},{inner|outer}.csv`

In the experiment script, set `target` to the dataset name (e.g. `monash_m1_quarterly`). Uncomment the appropriate loader (`ChronosDataset` vs `LongHorizonDatasetR`) for that dataset. Models are selected via `ModelsConfig.MODEL_CLASSES` in [`src/neural/nf_arch.py`](src/neural/nf_arch.py).

```bash
cd scripts/experiments
python experiments-run-all.py
```

**Models used in the paper analyses:** KAN, MLP, NHITS, TFT, PatchTST, GRU, Informer.

**Datasets:** `monash_tourism_quarterly`, `monash_tourism_monthly`, `monash_m3_quarterly`, `monash_m3_monthly`, `monash_m1_monthly`, `monash_m1_quarterly`, `TrafficL`, `ECL`.

### 1. Learning curve analysis

[`scripts/analysis/2_learning_curve.py`](scripts/analysis/2_learning_curve.py)

For each dataset and model, repeatedly samples `k` inner configs (`LEARNING_CURVE` values), selects the best by inner error, and records outer MASE. Produces per-dataset plots and an average-rank curve.

```bash
python scripts/analysis/2_learning_curve.py
```

Outputs: `assets/outputs/rs_lc,{dataset}.pdf` (and `rs_lc,average_rank.pdf`).

Useful flags at the top of the script: `N_REPS`, `SHOW_UNCERTAINTY_BANDS`, `DROP_OUTLIER_SCORES` / `MAX_SCORE`.

### 2. Overtuning analysis

[`scripts/analysis/3_overtuning.py`](scripts/analysis/3_overtuning.py)

Builds search trajectories of length `TRAJECTORY_SIZE` to study how validation-selected configs behave on the outer evaluation.

```bash
python scripts/analysis/3_overtuning.py
```

Outputs: `assets/outputs/overtuning,{dataset}.pdf`.

### 3. Hyperparameter importance analysis

[`scripts/analysis/4_hyperparams_importance.py`](scripts/analysis/4_hyperparams_importance.py)

Runs fANOVA over the random-search outcomes to attribute performance variance to hyperparameters (per model, aggregated across datasets).

```bash
python scripts/analysis/4_hyperparams_importance.py
```

Outputs: `assets/outputs/hp_importance,{model}.pdf` and `assets/outputs/hp_importance_summary.csv`.

### 4. Metalearning-based search

**Run search** — [`scripts/analysis/5_at.py`](scripts/analysis/5_at.py)

Leave-one-dataset-out metalearning / active-testing selection (`SEARCH_METHOD = 'AT'`), compared with random search (`RS`) baselines for a given trial budget `N_TRIALS`.

```bash
python scripts/analysis/5_at.py
```

Re-run with different `N_TRIALS` values as needed (paper analyses use budgets such as 5, 25, 100). Outputs CSV files under `assets/results/` (e.g. `search,{N_TRIALS},{model}.csv`).

**Summarize** — [`scripts/analysis/5_at_analysis.py`](scripts/analysis/5_at_analysis.py)

Aggregates those CSVs across models and budgets (including differences vs MLP) and prints a LaTeX table.

```bash
python scripts/analysis/5_at_analysis.py
```

