# AirBench

**AirBench** is a large-scale benchmark dataset and evaluation suite for short-term air quality forecasting using Time Series Foundation Models (TSFMs). It provides a standardised, reproducible protocol for evaluating TSFMs in a zero-shot setting across multiple countries, pollutants, and monitoring networks.

---

## Background

Air quality forecasting is a critical public health problem, yet existing time series benchmarks are dominated by financial, energy, and traffic domains. TSFMs have shown strong zero-shot generalisation across many forecasting tasks, but their performance on environmental monitoring data — characterised by high missingness, multi-pollutant heterogeneity, and cross-country distributional shift — remains poorly understood.

AirBench addresses this gap by introducing a multi-country, multi-pollutant benchmark built from reference-grade ground station data. Rather than proposing a new model, the contribution is the dataset, the evaluation infrastructure, and the empirical findings from running a suite of TSFMs and baselines under a consistent protocol.

---

## Dataset

Data is sourced from four national air quality monitoring networks:

| Network | Country | Agency |
|---------|---------|--------|
| AQS | United States | EPA |
| CPCB | India | Central Pollution Control Board |
| CNEMC | China | China National Environmental Monitoring Centre |
| AURN | United Kingdom | Automatic Urban and Rural Network |

**Pollutants:** PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃

**Time range:** January 2023 – December 2025 (3 years, hourly resolution)

**Evaluation protocol:** 2-month context window, 168-hour (1-week) prediction horizon, biweekly sliding window stride.

---

## Models Evaluated

**TSFMs (zero-shot):** Chronos, Moirai, TTM, Lag-Llama, Kairos, and others

**Baselines:** DeepAR (probabilistic ML), ARIMA (statistical), Seasonal Naïve

**Metrics:** MASE (point accuracy), CRPS (probabilistic calibration)

---

## Repository Structure

```
airbench/
├── pyproject.toml          # unified dependencies
├── config.yaml             # paths and dataset configuration
│
├── data/                   # download, process, impute, visualise
│   ├── download.py
│   ├── process.py
│   ├── impute.py
│   └── visualise.py
│
├── eval/                   # evaluation harness (based on TIME framework)
│   ├── src/
│   └── experiments/
│
└── scripts/                # end-to-end runner scripts
```

The evaluation harness is adapted from the [TIME framework](https://github.com/zqiao11/TIME).

---

## Setup

```bash
# Create environment (Kairos requires a separate venv due to transformers conflicts)
uv venv
uv pip install -e .

# For Kairos specifically
uv venv envs/kairos
uv pip install -e ".[kairos]" --python envs/kairos
```

---
## Data section:
#

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