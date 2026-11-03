# CS5998-capstone-project

Master's in DS and AI Capstone Project (CS5998), University of Moratuwa — Predicting credit default from customer statement histories.

## What this project asks

Credit-risk models in industry usually compress a customer's monthly statements into summary numbers (latest balance, average spend, count of late payments) and train a gradient-boosted tree on them. That throws away the order in which things happened: a customer whose payments slip a little more each month looks the same as one with a single bad month if their averages match.

This project tests whether that ordering carries signal. On the American Express default-prediction data (up to 13 monthly statements per customer), it compares a tuned LightGBM on aggregated features against sequence models (GRU, Transformer) that read the statements in order, and measures what any gain costs in complexity and explainability.

## Status

| | |
|---|---|
| Submission 01 — Project Definition | Submitted ([document](reports/Submission_01_Project_Definition.pdf)) |
| Submission 02 — Technical Checkpoint | In progress (due 27 November 2026) |
| Submission 03 — Final Submission | Not started |

## Data

| Dataset | Role | Source |
|---|---|---|
| American Express – Default Prediction | Labels, and the reference copy of the raw statements | [Kaggle competition](https://www.kaggle.com/competitions/amex-default-prediction) |
| AMEX data – integer dtypes, Parquet | Working copy of the statements used by the pipeline | [raddar on Kaggle](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) |

The data is not committed to this repository, and the competition rules do not allow redistributing it. Downloading needs a Kaggle account that has accepted the competition rules, and an API token (see Setup).

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

The data is about 5 GB for the working copy, plus the labels. By default it goes in `data/` inside the repository, which is git-ignored. If the repository sits in a synced folder (OneDrive, Dropbox), point the scripts somewhere else instead:

```bash
set AMEX_DATA_DIR=C:\capstone-data         # Windows (cmd)
$env:AMEX_DATA_DIR = "C:\capstone-data"    # Windows (PowerShell)
export AMEX_DATA_DIR=~/capstone-data       # macOS / Linux
```

To download, put a Kaggle API token (Kaggle → Settings → API → Create New Token) at `~/.kaggle/kaggle.json`, accept the [competition rules](https://www.kaggle.com/competitions/amex-default-prediction/rules), then run:

```bash
python scripts/download_data.py
```

## Results

Not yet available. Populated as experiments are run.
