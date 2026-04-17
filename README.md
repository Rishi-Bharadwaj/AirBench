# AirBench

**AirBench** is a large-scale benchmark dataset and evaluation suite for short-term air quality forecasting using Time Series Foundation Models (TSFMs). It provides a standardised, reproducible protocol for evaluating TSFMs in a zero-shot setting across multiple countries, pollutants, and monitoring networks.

---

## Background

Air quality forecasting is a critical public health problem, yet existing time series benchmarks are dominated by financial, energy, and traffic domains. TSFMs have shown strong zero-shot generalisation across many forecasting tasks, but their performance on environmental monitoring data — characterised by high missingness, multi-pollutant heterogeneity, and cross-country distributional shift — remains poorly understood.

AirBench addresses this gap by introducing a multi-country, multi-pollutant benchmark built from reference-grade ground station data. Rather than proposing a new model, the contribution is the dataset, the evaluation infrastructure, and the empirical findings from running a suite of TSFMs and baselines under a consistent protocol.

---

## Dataset

Data is sourced from seven national and regional air quality monitoring networks:

| Network | Country / Region | Agency |
|---------|-----------------|--------|
| EPA (AQS) | United States | Environmental Protection Agency |
| CPCB | India | Central Pollution Control Board |
| CNEMC | China | China National Environmental Monitoring Centre |
| AURN | United Kingdom | Automatic Urban and Rural Network |
| EEA (France) | France | European Environment Agency |
| EEA (Germany) | Germany | European Environment Agency |
| SINAICA | Mexico | Sistema Nacional de Información de la Calidad del Aire |

**Pollutants:** PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃

**Time range:** July 2022 – June 2025 (3 years, hourly resolution)

**Evaluation protocol:** 168-hour (1-week) context window, 24-hour prediction horizon, evaluated over a 2-year rolling test period.

**Imputation:** Stations with more than 30% missing data are excluded. Gaps of up to 336 hours (2 weeks) are filled by linear interpolation; longer gaps are left as-is and handled by the evaluation harness.

---

## Models Evaluated

**TSFMs (zero-shot):** Chronos-Bolt, Chronos 2, Moirai, Moirai 2, TimesFM 1.0 / 2.0 / 2.5, TiRex, VisionTS++, Sundial, Kairos

**Supervised baselines:** DLinear, PatchTST (trained per-dataset)

**Classical baselines:** Seasonal Naïve, AutoARIMA, AutoETS

**Metrics:** MASE (point accuracy) and CRPS (probabilistic calibration), both normalised by Seasonal Naïve.

---

## Repository Structure

```
airbench/
├── pyproject.toml              # unified dependencies for the data pipeline
│
├── data/
│   ├── data.yaml               # paths and config for all data pipeline stages
│   ├── data_download_scripts/  # notebooks/scripts to generate download URLs
│   │   ├── epa_download.ipynb
│   │   ├── cpcb_download.ipynb
│   │   ├── cnemc_download.ipynb
│   │   ├── aurn_download.ipynb
│   │   └── openaq_download.ipynb
│   ├── data_preprocess_scripts/ # per-network preprocessing (raw → cleaned CSVs)
│   │   ├── epa_preprocess.py
│   │   ├── cpcb_preprocess.py
│   │   ├── cnemc_preprocess.py
│   │   ├── aurn_preprocess.py
│   │   ├── eea_preprocess.py
│   │   ├── sinaica_preprocess.py  # also handles download via Rsinaica API
│   │   └── openaq_preprocess.py
│   ├── imputation.py           # filters stations and fills short gaps
│   ├── visualise.py            # heatmap visualisation of coverage and missingness
│   └── aq_dataset_builder.py  # converts imputed CSVs → HuggingFace Arrow datasets
│
└── eval/
    ├── config.yaml             # model list, dataset paths, leaderboard settings
    ├── datasets.yaml           # test/val lengths and prediction horizon per dataset
    ├── run.py                  # main evaluation runner
    ├── compute_local_leaderboard.py  # aggregates results into ranked tables
    ├── leaderboard_helpers.py  # normalisation and consistency checks
    ├── leaderboard_utils.py    # display utilities, model group definitions
    ├── src/timebench/          # evaluation harness (adapted from TIME framework)
    └── experiments/            # one script per model
        ├── chronos_bolt.py
        ├── chronos2.py
        ├── moirai.py / moirai2.py
        ├── timesfm1.0.py / timesfm2.0.py / timesfm2.5.py
        ├── tirex_model.py
        ├── visiontspp.py
        ├── sundial.py
        ├── kairos_model.py
        ├── dlinear.py
        ├── patchtst.py
        ├── seasonal_naive.py
        └── auto_ets.py
```

The evaluation harness is adapted from the [TIME framework](https://github.com/zqiao11/TIME).

---

## Data Pipeline

Each network goes through the same four stages, all configured via `data/data.yaml`:

1. **Download** — download scripts generate lists of URLs (or use an API scraper for SINAICA) which are then fetched with `aria2c`. EEA data can be bulk-downloaded directly from the EEA data hub. SINAICA has one combioned script for scarping and preprocessing.

2. **Preprocess** — per-network scripts convert raw files to a unified format: one CSV per station per pollutant, all timestamps converted to local time with unit normalisation applied.

3. **Visualise** — `visualise.py` produces heatmaps showing data coverage over time for each station, useful for inspecting gaps before imputation.

4. **Impute** — `imputation.py` drops stations below the 70% completeness threshold and fills remaining gaps (up to 336 hours) via linear interpolation.

After imputation, `aq_dataset_builder.py` converts the per-station CSVs into HuggingFace Arrow datasets (one per network) ready for the evaluation harness.

---

## Evaluation

The evaluation harness is driven by `eval/config.yaml`. It is adapted from the TIME leaderboard, (https://github.com/zqiao11/TIME). To run evaluations:

```bash
cd eval
python run.py
```

Models, datasets, and hyperparameters are all declared in `config.yaml`. Each experiment script in `experiments/` is self-contained and can also be run directly.

To compute the leaderboard from saved results:

```bash
cd eval
python compute_local_leaderboard.py
```

Results are aggregated across datasets and normalised by Seasonal Naïve, with separate per-pollutant breakdowns.

---

## Setup

```bash
# Create environment and install data pipeline dependencies
uv venv
uv pip install -e .

# Kairos requires a separate venv due to transformers version conflicts
uv venv envs/kairos
uv pip install -e ".[kairos]" --python envs/kairos
```

Most eval dependencies (Chronos, Moirai, TimesFM, etc.) are installed on-demand by `run.py` per model, as declared in `config.yaml`.

---

## Citation

If you use AirBench in your work, please cite:

```bibtex
@misc{airbench2025,
  title   = {AirBench: Benchmarking Time Series Foundation Models on Multi-Country Air Quality Forecasting Data},
  author  = {TODO},
  year    = {2025},
}
```

Please also cite the [TIME framework](https://github.com/zqiao11/TIME) which the evaluation harness is based on.

---

## License

Dataset redistribution is subject to the terms of each source network. See `data/LICENSES.md` for details.