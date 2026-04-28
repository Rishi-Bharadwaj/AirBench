# Data Pipeline

All scripts are configured via `data.yaml`, which contains paths and parameters for each network and each pipeline stage. **Run all scripts from the project root** (`AtmoBench/`), since paths in `data.yaml` are relative to it.

### Example

```bash
# from the project root
python data/data_preprocess_scripts/sinaica_preprocess.py data/data.yaml
python data/visualise.py data/data.yaml sinaica
python data/imputation.py data/data.yaml sinaica
python data/aq_dataset_builder.py --config data/data.yaml --dataset SINAICA/H
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

### EPA (`epa_preprocess.py`)

Input is one ZIP per year per pollutant from the EPA AQS bulk download, named by AQS parameter code (e.g. `hourly_42401_2022.zip` for SO2). The script runs in two steps:

1. **Split** — unpacks the large national CSVs and writes one file per site per year, identified by `(State Code, County Code, Site Num)`.
2. **Join** — concatenates years per site, reindexes to the target date range, and converts units.

Unit conversions applied (molecular-weight-based, standard atmosphere):

| Pollutant | Raw unit | Factor | Output unit |
|-----------|----------|--------|-------------|
| SO2 | ppb | ×2.62 | µg/m³ |
| O3 | ppm | ×1960 | µg/m³ |
| NO2 | ppb | ×1.88 | µg/m³ |
| CO | ppm | ×1.15 | mg/m³ |
| PM2.5, PM10 | µg/m³ | ×1.0 | µg/m³ |

Output filenames: `site_{state}_{county}_{sitenum}_{FORMULA}.csv`.

### CPCB (`cpcb_preprocess.py`)

Input is a single ZIP containing per-site CSVs at 15-minute resolution. Data are already in standard units (columns are named `PM2.5 (µg/m³)` etc.) so no unit conversion is needed. The script groups files by site, concatenates across years, and resamples to hourly by taking the **median** of each 15-minute window. Output filenames drop the `_15Min` suffix from the site stem.

### CNEMC (`cnemc_preprocess.py`)

Input is one CSV per day (named `YYYYMMDD.csv`) in a wide format where each column is a station and each row is a pollutant×hour combination. All daily files are concatenated into a single in-memory DataFrame, then split out by site and pollutant. Data are already in µg/m³ (PM, NO2, SO2, O3) and mg/m³ (CO) — no conversion needed.

A station list spreadsheet (`站点列表-2022.02.13起.xlsx`) is required to map station codes to metadata. This is available on Zenodo in our dataset. `china_sampling.ipynb` can be used to create a random subsample of stations for faster iteration.

### AURN (`aurn_preprocess.py`)

Input is `.RData` files (one per site per year) downloaded from the AURN data archive. The `rdata` Python package parses these without requiring an R installation. Timestamps are stored as Unix seconds UTC inside the RData objects; the script converts them to `Europe/London` local time and strips timezone info. Processing is two-stage: RData → intermediate CSVs, then intermediate CSVs → final per-pollutant output. Data are already in µg/m³ / mg/m³, no unit conversion needed.

Set `rdata_conversion: false` in `data.yaml` to skip the RData→CSV step if you've already done it.

### EEA (`eea_preprocess.py`)

Input is the bulk Parquet download from the EEA Air Quality Data Hub (which covers all European countries). The script filters to France (`FR`) and Germany (`DE`) and maps EEA numeric pollutant IDs to names:

| EEA ID | Pollutant |
|--------|-----------|
| 1 | SO2 |
| 5 | PM10 |
| 7 | O3 |
| 8 | NO2 |
| 10 | CO |
| 6001 | PM2.5 |

Processing is two-stage: Parquet → filtered CSVs (one per sampling point), then group by site+pollutant and filter to rows with `Validity >= 1`. Data are already in the correct units. Output is written to `{base_out_dir}/FR/processed/` and `{base_out_dir}/DE/processed/`.

Set `convert_parquet: false` in `data.yaml` to skip the first stage if already done.

### SINAICA (`sinaica_preprocess.py`)

Unlike the other networks, SINAICA has no bulk download API, so this script handles both download and preprocessing. It scrapes `sinaica.inecc.gob.mx` directly, modelled on the R package `Rsinaica`. The station list is fetched dynamically from the site's HTML. For each station × pollutant × year, the script POSTs to the data endpoint and extracts the embedded JavaScript array (`var dat = [...]`) from the response. Requests are rate-limited (0.3 s between calls) with exponential-backoff retries.

Unit conversions (gas-phase pollutants are reported in ppm by SINAICA):

| Pollutant | Raw unit | Factor | Output unit |
|-----------|----------|--------|-------------|
| O3 | ppm | ×1960 | µg/m³ |
| NO2 | ppm | ×1880 | µg/m³ |
| SO2 | ppm | ×2620 | µg/m³ |
| CO | ppm | ×1.15 | mg/m³ |
| PM2.5, PM10 | µg/m³ | ×1.0 | µg/m³ |

Only rows with `val == 1` (the site's own validity flag) are kept. The script is **resumable**: it skips any output file that already exists, so an interrupted run can be restarted safely. It also saves `sinaica_stations.csv` (the full station list) to the output directory.

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
python data/aq_dataset_builder.py --config data/data.yaml --dataset SINAICA/H
```

Reads the `sources` section of `data.yaml`, which maps dataset names to imputed CSV directories and output paths. Converts each directory of per-station CSVs into a HuggingFace Arrow dataset (univariate series format) readable by the evaluation harness.

Each series is identified by an `item_id` of the form `<site_id>_<POLLUTANT>`, which the leaderboard uses to aggregate results per pollutant.

Output is written to `data/hf_dataset/<NETWORK>/H/` and pointed to by `eval/config.yaml`.
