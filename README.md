#Smart Meter Data Analytics Pipeline

A **Python pipeline for smart-meter data ingestion, load-profile feature engineering, customer segmentation, and non-technical-loss / anomaly detection**, modeled on the Irish CER Smart Metering Trial, with an interactive Streamlit dashboard for exploring the results.

It combines four stages into one pipeline:

- **Ingestion & data-quality cleaning** -: synthetic CER-shaped data generation for demos, plus a drop-in loader/cleaner for the real ISSDA CER trial files.
- **Load-profile feature engineering** -: per-meter load factor, time-of-use consumption ratios, and within-window behavior-change features.
- **Customer segmentation** -: K-Means clustering of meters into load-profile archetypes.
- **Anomaly / non-technical-loss detection** -: Isolation Forest flagging of meters with suspicious consumption patterns (bypass, tamper, faults).

---

## Table of Contents

- [Video](#Video)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Mathematical Formulation](#mathematical-formulation)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Inputs What Each Module Expects](#inputs--what-each-module-expects)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [License](#license)

---
## Video
**Smart Meter Analytics Dashboard** load-profile clustering (K-Means), anomaly/theft flagging (Isolation Forest), and the injected-tamper-event validation demo:

![Smart Meter Dashboard demo](assets/Dashboard.gif)
```
```
## Project Structure

```
Smart-Meter-Data-Analytics-Pipeline/
├── main.py                     # CLI entry point/pipeline orchestrator
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── src/
│   ├── ingestion.py            # Synthetic data generation + real CER file loading/cleaning
│   ├── features.py             # Load-profile feature extraction
│   ├── segmentation.py         # K-Means customer clustering
│   └── anomaly.py              # Isolation Forest anomaly detection
├── test_ingestion.py           # Stage 1 smoke test
├── test_features.py            # Stage 2 smoke test
├── test_segmentation.py        # Stage 3 smoke test
├── test_anomaly.py             # Stage 4 smoke test
├── test_cer_migration.py       # Unit tests for CER timecode decoding + real-data cleaning
├── data/
│   ├── raw/                    # Generated/loaded raw meter reads (gitignored)
│   └── processed/              # Feature/segmentation/anomaly outputs (gitignored)
├── models/                     # Trained kmeans/scaler/iso_forest artifacts (gitignored)
├── .gitignore
└── README.md
```

---

## How It Works

**1. Ingestion** (`src/ingestion.py`)
`generate_raw_cer_dataset` synthesizes half-hourly (48-interval/day) consumption for a mix of Residential, Night/EV-Heavy, and Commercial load profiles, matching the Irish CER trial's schema. `load_real_cer_dataset` is a drop-in replacement for real ISSDA CER files: it decodes the trial's 5-digit day+halfhour timecodes (`decode_cer_timecode`), merges in per-meter Residential/SME allocations, and runs a deliberately narrow data-quality pass (`clean_cer_data`) that only removes definite errors -: duplicate reads and negative kwh -: without touching statistical outliers, since that's the anomaly detector's job downstream.

**2. Feature engineering** (`src/features.py`)
`extract_load_features` groups raw reads by meter and computes load factor, time-of-use consumption ratios (night / morning-peak / evening-peak, on Irish-grid-standard interval definitions), a weekend-vs-weekday ratio, and two within-window behavior-change features -: `half_period_ratio` and `daily_consumption_std` -: that specifically capture a mid-window change in behavior (e.g. a bypass event), which whole-window averages structurally cannot see.

**3. Segmentation** (`src/segmentation.py`)
`train_customer_clusters` standardizes the five core load features and clusters meters into load-profile archetypes with K-Means, scored with silhouette score. Trained `KMeans`/`StandardScaler` artifacts are saved for reuse.

**4. Anomaly detection** (`src/anomaly.py`)
`detect_meter_anomalies` fits an Isolation Forest over the same five clustering features plus the two within-window behavior-change features, flagging meters whose combination of load shape *and* mid-window stability looks like non-technical loss rather than normal usage variation.

**5. Dashboard** (`app.py`)
Reads the pipeline's raw and final-analytics CSVs, merges them, and renders three tabs -: Customer Load Profiling (diurnal load curves by cluster), Anomaly & Theft Flagging (flagged-meter table + time series), and Single Meter Drill-Down. A sidebar control can re-run the full pipeline with a known, injected ground-truth tamper event to demonstrate that the temporal features actually catch it.

---

## Mathematical Formulation

### Load-profile features (`src/features.py`)

**Load factor** (average load relative to peak load):

$$LF = \frac{\overline{kWh}}{kWh_{max}}$$

**Time-of-use consumption ratios**, for TOU window $w \in \{\text{night, morning-peak, evening-peak}\}$:

$$r_w = \frac{\sum_{t \in w} kWh_t}{\sum_{t} kWh_t}$$

**Weekend ratio:**

$$R_{weekend} = \frac{\overline{kWh}_{weekend}}{\overline{kWh}_{weekday}}$$

**Half-period ratio** (within-window behavior-change signature -: the second half of the observation window vs. the first):

$$R_{half} = \frac{\overline{kWh}_{2^{nd}\,half}}{\overline{kWh}_{1^{st}\,half}}$$

**Daily consumption volatility** (standard deviation of each meter's daily totals across the observation window):

$$\sigma_{daily} = \text{std}\big(\{\textstyle\sum_{t \in d} kWh_t : d \in \text{days}\}\big)$$

### CER timecode decoding (`decode_cer_timecode`)

The real CER trial encodes each read as a 5-digit `day_code || halfhour_code`:

$$\text{day\_code} = \left\lfloor \frac{\text{timecode}}{100} \right\rfloor, \qquad \text{halfhour\_code} = \text{timecode} \bmod 100$$

$$\text{timestamp} = \text{epoch\_date} + (\text{day\_code} - 1)\text{ days} + 30(\text{halfhour\_code} - 1)\text{ minutes}, \quad \text{halfhour\_code} \in [1, 48]$$

### K-Means segmentation (`src/segmentation.py`)

Features are standardized before clustering:

$$z = \frac{x - \mu}{\sigma}$$

K-Means assigns cluster membership to minimize total within-cluster variance:

$$\underset{C}{\arg\min} \sum_{k=1}^{K} \sum_{x \in C_k} \lVert x - \mu_k \rVert^2$$

Cluster separation is scored per point with the silhouette coefficient:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ is the mean distance from point $i$ to other points in its own cluster, and $b(i)$ is the mean distance to points in the nearest other cluster.

### Isolation Forest anomaly detection (`src/anomaly.py`)

Anomaly score for a point $x$, based on average path length $E[h(x)]$ to isolate it across an ensemble of random trees:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}, \qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

where $H(i)$ is the harmonic number and $c(n)$ normalizes for sample size $n$. Points that are isolated in unusually few splits (short average path length) score closer to anomalous; the `contamination` parameter sets what fraction of meters are labeled `is_anomaly` (default 5%, a modeling assumption rather than a measured non-technical-loss rate -: see [Roadmap](#roadmap)).

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip / venv

### Installation

```bash
git clone https://github.com/manusri4002/Smart-Meter-Data-Analytics-Pipeline.git
cd Smart-Meter-Data-Analytics-Pipeline

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Project

**1. Run the core pipeline** (generates synthetic data on first run, extracts features, trains the cluster + anomaly models, saves the final analytics CSV):

```bash
python main.py
```

Useful flags:

```bash
python main.py --num-meters 100 --days 30
python main.py --force-regenerate
python main.py --inject-tamper-events   # injects a known ground-truth tamper
                                         # event to validate the anomaly detector
```

**2. Launch the dashboard** (requires the pipeline output to already exist):

```bash
streamlit run app.py
```

**3. Run the tests:**

```bash
python test_ingestion.py
python test_features.py
python test_segmentation.py
python test_anomaly.py
python test_cer_migration.py
```

---

## Inputs What Each Module Expects

### Ingestion (`src/ingestion.py`)

**Synthetic generation** (`generate_raw_cer_dataset`):

| Field | Type | Example | Notes |
|---|---|---|---|
| `num_meters` | int | `50` | Number of synthetic meters |
| `days` | int | `14` | Observation window length |
| `seed` | int | `42` | RNG seed for reproducibility |
| `inject_tamper_events` | bool | `False` | Drops consumption to near-zero for the 2nd half of the window on the first `n_tamper_events` meters |
| `n_tamper_events` | int | `2` | Number of meters to tamper |

**Real CER trial files** (`load_real_cer_dataset`), positional/unlabeled columns:

| Field | Type | Example | Notes |
|---|---|---|---|
| `meter_id` | int | `1401` | Meter identifier |
| `timecode` | int (5-digit) | `19525` | `day_code` (day 1 = epoch) + `halfhour_code` (1-48) |
| `kwh` | float | `0.842` | Interval consumption |

### Feature Engineering (`src/features.py`)

Expects the canonical schema produced by ingestion: `meter_id`, `timestamp`, `kwh`. No additional inputs.

### Segmentation (`src/segmentation.py`)

| Field | Type | Example |
|---|---|---|
| `n_clusters` | int | `3` |
| `random_state` | int | `42` |

Requires the feature columns `mean_kwh`, `load_factor`, `night_ratio`, `morn_peak_ratio`, `eve_peak_ratio` (`CLUSTER_FEATURE_COLS`) to already exist on the input DataFrame.

### Anomaly Detection (`src/anomaly.py`)

| Field | Type | Example |
|---|---|---|
| `contamination` | float | `0.05` |
| `random_state` | int | `42` |

Requires `CLUSTER_FEATURE_COLS` plus `half_period_ratio` and `daily_consumption_std` to already exist on the input DataFrame.

### Dashboard (`app.py`)

Reads `data/raw/cer_raw_data.csv` and `data/processed/cer_final_analytics.csv` -: both produced by `python main.py`. No manual inputs beyond the sidebar cluster/anomaly filters and the demo tamper-event button.

---

## Outputs

- **Ingestion:** canonical `meter_id` / `timestamp` / `kwh` raw CSV, optionally merged with segment allocations.
- **Feature engineering:** per-meter feature table -: `tot_kwh`, `mean_kwh`, `max_kwh`, `std_kwh`, `load_factor`, `night_ratio`, `morn_peak_ratio`, `eve_peak_ratio`, `weekend_ratio`, `half_period_ratio`, `daily_consumption_std`.
- **Segmentation:** the above plus `cluster` and `silhouette_score`.
- **Anomaly detection:** the above plus `anomaly_score` (-1/1) and `is_anomaly` (bool) -: this is the final `cer_final_analytics.csv` the dashboard reads.

---

## Tech Stack

- **Frontend:** Streamlit + Plotly
- **ML:** scikit-learn (`KMeans`, `IsolationForest`, `StandardScaler`, `silhouette_score`)
- **Data:** pandas, NumPy
- **Model persistence:** joblib
- **Data source:** Irish CER Smart Metering Trial (ISSDA), synthetic generator for demo/dev use

---

## Roadmap

- [ ] Verify `CER_TRIAL_EPOCH_DATE` against the real ISSDA CER codebook -: currently a documented placeholder assumption
- [ ] Vectorize `decode_cer_timecode`'s row-wise `.apply()` call for real-trial scale (~5,000 meters, months of data)
- [ ] Tune the Isolation Forest `contamination` default (currently 0.05, a modeling assumption) against real labeled non-technical-loss data once available
- [ ] Add authentication/rate-limiting to the dashboard's "Run pipeline" button before any shared/public deployment, since it currently overwrites the shared output files with no confirmation
- [ ] Add unit tests for `src/segmentation.py` and `src/anomaly.py` at the same rigor as `test_cer_migration.py`
- [ ] Formalize the CSV schemas (raw + final analytics) as an explicit documented data contract rather than relying on the dashboard's runtime column-presence check

---

## License

No license has been chosen yet
