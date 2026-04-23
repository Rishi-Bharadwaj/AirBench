# Data Pipeline

All scripts are configured via `data.yaml`, which contains paths and parameters for each network and each pipeline stage. **Run all scripts from the project root** (`AtmoBench/`), since paths in `data.yaml` are relative to it.

### Example

```bash
# from the project root
python data/data_preprocess_scripts/aurn_preprocess.py data/data.yaml
python data/visualise.py data/data.yaml aurn
python data/imputation.py data/data.yaml aurn
python data/aq_dataset_builder.py --config data/data.yaml --dataset AURN/H
```

---

## Networks

| Key in `data.yaml` | Network | Country |
|---------------------|---------|---------|
| `epa` | EPA AQS | United States |
| `cpcb` | CPCB | India |
| `cnemc` | CNEMC | China |
| `aurn` | AURN | United Kingdom |
| `eea_fr` / `eea_de` | EEA | France / Germany |
| `sinaica` | SINAICA | Mexico |

All networks produce the same output format: one CSV per station per pollutant, with columns `Timestamp` (local time, hourly) and the pollutant value.

---

## Stage 1 — Download

### EPA, CPCB, CNEMC, AURN

The notebooks in `data_download_scripts/` generate lists of URLs for each network. Download them with `aria2c`:

```bash
aria2c -i urls.txt -j 16 -x 16
```

### EEA (France and Germany)

Bulk download directly from the [EEA Air Quality Data Hub](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d). Click "Air quality data download service" and follow the instructions. The download includes data for all European countries; `eea_preprocess.py` filters to France and Germany.

### SINAICA

SINAICA does not have a public bulk download API. `data_preprocess_scripts/sinaica_preprocess.py` handles both download and preprocessing together using a custom scraper modelled on the R package `Rsinaica`.

### OpenAQ (not used in the paper)

Scripts are included for completeness. Requires a free OpenAQ API key and downloads from their AWS S3 bucket.

---

## Stage 2 — Preprocess

One script per network in `data_preprocess_scripts/`. Each script:

- Parses the raw files for that network's format
- Converts all timestamps to **local time** (no UTC offset retained)
- Applies **unit conversions** to a standard set of units (µg/m³ for most pollutants, mg/m³ for CO)
- Writes one CSV per station per pollutant: `<site_id>_<POLLUTANT>.csv`

Configure input/output paths in the relevant section of `data.yaml` before running.

**AURN note:** Raw files are in R's `.RData` format. `aurn_preprocess.py` uses the `rdata` Python package to convert them to DataFrames without requiring an R installation.

**CNEMC note:** A station list spreadsheet (`站点列表-2022.02.13起.xlsx`) is required to map station codes to metadata. This is available on zenodo in our dataset.
`china_sampling.ipynb` can be used to create a random subsample of stations for faster iteration.

---

## Stage 3 — Visualise

```bash
python data/visualise.py data/data.yaml <network_key>
```

Produces a heatmap for each pollutant showing all stations on the y-axis and time on the x-axis. Colour encodes concentration; white cells are missing data. Stations are sorted by amount of missing data so gaps are easy to spot.

Output images are saved to the `image_dir` path in `data.yaml`. The script also saves per-pollutant DataFrames (one column per station) to `vis_dicts/` — these are reused by the imputation stage to compute per-station missingness statistics without re-reading all individual CSVs.

---

## Stage 4 — Impute

```bash
python data/imputation.py data/data.yaml <network_key>
```

Two-step filtering and gap-filling:

**1. Station filtering** (uses the `vis_dicts/` DataFrames from Stage 3):
- Stations with more than `max_data_missing`% missing data overall are dropped.
- Stations with any single contiguous gap longer than `max_gap_hours` hours are dropped.

Default thresholds (set in `data.yaml`): `max_data_missing: 30`, `max_gap_hours: 336` (2 weeks).

**2. MSTL imputation** (applied to each surviving station × pollutant series):
- Fits a Multi-Seasonal-Trend decomposition (MSTL) with daily (24 h) and weekly (168 h) periods using robust LOESS.
- Performs linear interpolation in the **deseasonalised** space, then adds the seasonal component back. This produces smoother, more physically plausible fills than raw linear interpolation.
- Imputed values are clipped to `[0, observed_max]`.
- Negative raw values are treated as missing before imputation.

Runs in parallel across stations using `ProcessPoolExecutor` (`max_workers` set in `data.yaml`).

---

## Stage 5 — Build HuggingFace Dataset

```bash
python data/aq_dataset_builder.py --config data/data.yaml
# or a specific network:
python data/aq_dataset_builder.py --config data/data.yaml --dataset CPCB/H
```

Reads the `sources` section of `data.yaml`, which maps dataset names to imputed CSV directories and output paths. Converts each directory of per-station CSVs into a HuggingFace Arrow dataset (univariate series format) readable by the evaluation harness.

Each series is identified by an `item_id` of the form `<site_id>_<POLLUTANT>`, which the leaderboard uses to aggregate results per pollutant.

Output is written to `data/hf_dataset/<NETWORK>/H/` and pointed to by `eval/config.yaml`.
