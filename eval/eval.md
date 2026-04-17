# Evaluation

The eval harness is adapted from the [TIME framework](https://github.com/zqiao11/TIME). Everything runs from the `eval/` directory.

---

## Configuration

### `config.yaml`

The main config file. Controls which models and datasets to run and where results are written.

```yaml
time_repo: .                        # path to the eval directory (contains experiments/)
data_dir: ./data/hf_dataset         # HuggingFace Arrow datasets produced by aq_dataset_builder.py
datasets_config: datasets.yaml      # per-dataset test lengths and prediction horizon
```

Each model is declared as a block:

```yaml
models:
  chronos_bolt:
    script: experiments/chronos_bolt.py
    packages:
      - chronos-forecasting
    args:
      context_length: 168
      batch_size: 1024
```

- `script` — path to the experiment script relative to `time_repo`
- `packages` — pip packages installed into the active venv before the script runs
- `args` — CLI flags forwarded verbatim to the script (underscores → hyphens)
- `git_clone` (optional) — clones a repo before running (used by Kairos)

Most models are commented out in `config.yaml` by default. Uncomment the ones you want to run.

The `datasets` list at the bottom of `config.yaml` sets which datasets run by default (when no `dataset=` override is given).

### `datasets.yaml`

Defines the evaluation window for each dataset:

```yaml
datasets:
  CPCB/H:
    test_length: 17376    # number of hourly steps in the test split (≈ 2 years)
    val_length: 0
    short:
      prediction_length: 24   # 24-hour forecast horizon
```

The `short` horizon is the only one currently configured. `test_length: 17376` corresponds to approximately 2 years of hourly data minus one prediction window.

---

## Running Experiments

```bash
cd eval

# Run all enabled models on all default datasets
python run.py --config config.yaml

# Run a single model
python run.py --config config.yaml model=chronos_bolt

# Run a single dataset
python run.py --config config.yaml dataset=CPCB/H

# Run a specific model × dataset combination
python run.py --config config.yaml model=chronos_bolt dataset=CPCB/H
```

### What `run.py` does

1. Loads `config.yaml` and `datasets.yaml`.
2. Resolves which models and datasets to run (applying any `model=` / `dataset=` overrides).
3. For each model × dataset pair:
   - Installs the model's `packages` into the active venv via `uv pip install`.
   - Ensures `torch==2.10.0+cu128` is installed (reinstalls only if version differs).
   - Optionally `git clone`s a repo if `git_clone` is specified.
   - Launches the experiment script as a subprocess with `TIME_DATASET` set to `data_dir`.
4. Collects exit codes and prints a summary of any failures at the end.

The runner does not create separate venvs per model — all packages share the same venv. Kairos is the exception and requires a separate venv (see Setup in the main README).

### Results layout

Each experiment script writes its output to:

```
eval/output/results/<model_name>/<dataset_id>/<horizon>/
    metrics.npz     # per-series arrays: MASE, CRPS, MAE, RMSE (shape: [n_series, n_windows, ...])
    config.json     # evaluation config including item_ids (list of series identifiers)
```

`item_ids` encode the pollutant: each ID ends with `_<POLLUTANT>` (e.g. `site_105_..._CO`), which the leaderboard uses to separate results by pollutant.

---

## Computing the Leaderboard

```bash
cd eval

# Use defaults from config.yaml leaderboard section
python compute_local_leaderboard.py

# Override dataset or metric
python compute_local_leaderboard.py --dataset CPCB/H --metric CRPS
```

`seasonal_naive` results **must** be present in `output/results/` before running the leaderboard, as all metrics are normalised against it. Run `model=seasonal_naive` first if needed.

### Methodology

The leaderboard uses a **pollutant-balanced** aggregation to avoid datasets with many stations for a single pollutant dominating the ranking:

1. **Per-site mean** — average MASE/CRPS per series across all evaluation windows.
2. **Per-pollutant mean** — mean across sites within each pollutant, per (model, dataset, horizon).
3. **Per-dataset mean** — mean across pollutants, giving each pollutant equal weight.
4. **Normalise by Seasonal Naïve** — divide each model's score by Seasonal Naïve's score for the same (dataset, horizon). Values < 1 mean the model beats the baseline.
5. **Geometric mean across datasets** — final scalar ranking score.

Sites where the cross-model mean MASE or CRPS exceeds 50 are excluded from steps 2–5 to avoid outlier series distorting rankings.

### Outputs

All outputs are written to `output/leaderboard/`:

```
output/leaderboard/
    pollutant_balanced_leaderboard.csv       # overall ranking
    pollutant_balanced_leaderboard.tex       # LaTeX table for the paper
    per_dataset_horizon/                     # per-(dataset, horizon) normalized tables (.tex)
    per_dataset_horizon_csv/                 # same as CSVs
    per_pollutant/<DATASET>/<POLLUTANT>.tex  # per-pollutant tables
    per_pollutant_csv/<DATASET>/<POLLUTANT>.csv
```

---

## Model Groups

`leaderboard_utils.py` defines `MODEL_GROUPS` which maps model names to categories used to organise LaTeX tables:

| Category | Models |
|----------|--------|
| TSFMs | Chronos-Bolt, Chronos 2, Moirai, Moirai 2, TimesFM 1.0/2.0/2.5, TiRex, VisionTS++, Sundial, Kairos |
| ML Baselines | DLinear, PatchTST |
| Statistical Baselines | Seasonal Naïve, AutoETS |

Edit `MODEL_GROUPS` and `GROUP_ORDER` in `leaderboard_utils.py` when adding new models.

---

## Adding a New Model

1. Add an experiment script to `experiments/` following the pattern of existing scripts (they all accept `--dataset`, `--config`, `--context-length`, `--batch-size`).
2. Add a block to `config.yaml` under `models:` with the script path, required packages, and args.
3. Register the model in `MODEL_GROUPS` in `leaderboard_utils.py`.
4. Run it: `python run.py --config config.yaml model=<your_model>`.
