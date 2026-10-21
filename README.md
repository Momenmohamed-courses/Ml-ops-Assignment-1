# Breast Cancer Classifier: From Notebook to Production Python

first assignment for the Production AI Engineering course.
 The goal:   take simple ML experiment from a notebook and turn it into a clean, reproducible Python project that someone else can clone and run.

## What it does

It trains a model that predicts whether a breast tumor is **malignant** (cancerous) or **benign**, the model is saved to a file, and predictions are made by loading that file. No retraining needed.

## Dataset

used **Breast Cancer Wisconsin** dataset from scikit-learn.

- 569 patients, 30 numeric features, no missing values
- Labels: `0 = malignant`, `1 = benign`
- Slightly unbalanced: 357 benign (~63%) vs 212 malignant (~37%)

Because of that imbalance, data is splitted  in a stratified way.
## The ML pipeline

```
Load data → Inspect → Split train/test → Scale → Fit model → Evaluate → Save → Reload → Predict
```

- **Split:** 80% train / 20% test, stratified, with a fixed random seed so results are the same every run
- **Preprocessing:** `StandardScaler`. The features live on very different scales (area is in the hundreds, smoothness is around 0.1), and Logistic Regression learns better when they're comparable
- **No data leakage:** the scaler and the model sit together in a scikit-learn `Pipeline`, so the scaler only ever learns from the training data. The same fitted pipeline is saved and reused at prediction time

## Model choice

**Logistic Regression.** It's simple, fast, easy to explain, and a very solid baseline for a small numeric dataset like this one. The assignment didn't ask for tuning, so I kept the defaults apart from `max_iter=1000` to give it enough room to converge.

## Results

On the 114 test patients:

| Metric | Value |
|---|---|
| Accuracy | 0.982 |
| Precision (malignant) | 0.976 |
| Recall (malignant) | 0.976 |
| F1 (malignant) | 0.976 |

Confusion matrix (rows = real, columns = predicted):

|  | predicted malignant | predicted benign |
|---|---|---|
| **really malignant** | 41 | 1 |
| **really benign** | 1 | 71 |



## Project structure

```
MLOps-Assignment-1/
├── notebooks/
│   └── 01_exploration.ipynb   # my experiments: data checks, baseline model, results
├── production_app/
│   ├── __init__.py            # marks the folder as a package
│   ├── config.py              # reads settings from .env (with safe defaults)
│   ├── logging_config.py      # sets up logging once for the whole app
│   ├── data.py                # loads the dataset, splits it, reads prediction CSVs
│   ├── preprocessing.py       # the (unfitted) scaler
│   ├── training.py            # builds the pipeline and fits it
│   ├── evaluation.py          # calculates and logs the metrics
│   ├── model_service.py       # saves/loads the model, validates input, predicts
│   └── main.py                # the training and prediction workflows
├── examples/
│   └── sample_input.csv       # 3 test patients to try prediction with
├── models/                    # the trained model is saved here (not committed)
├── run.py                     # entry point: train or predict
├── .env.example               # the settings you can change
├── .pre-commit-config.yaml    # runs Ruff and Black before every commit
├── pyproject.toml / uv.lock   # dependencies, pinned for reproducibility
└── README.md
```

The notebook is my experimentation record.
The `production_app` package is the source of truth for the actual application.

## Getting started

### Prerequisites

- macOS or Linux
- Git
- [uv](https://docs.astral.sh/uv/) (it installs the right Python version for you, so you don't need Python preinstalled)

To install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal, or run `source $HOME/.local/bin/env`, so the `uv` command is found.

### 1. Clone and install

```bash
git clone https://github.com/Momenmohamed-courses/Ml-ops-Assignment-1.git
cd Ml-ops-Assignment-1
uv sync
```

`uv sync` creates the `.venv` and installs the exact versions from `uv.lock`.

### 2. Configure

```bash
cp .env.example .env
```

The defaults work out of the box. These are the settings you can change in `.env`:

| Variable | Default | What it does |
|---|---|---|
| `MODEL_PATH` | `models/pipeline.joblib` | where the trained model is saved and loaded from |
| `LOG_LEVEL` | `INFO` | how much logging to show (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `APP_ENV` | `development` | which environment the app runs in |
| `TEST_SIZE` | `0.2` | share of data kept for testing (must be between 0 and 1) |
| `RANDOM_SEED` | `42` | keeps the split the same on every run |
| `MAX_ITER` | `1000` | max training iterations for Logistic Regression |

If `.env` is missing, the app falls back to these defaults. A setting passed in the terminal (e.g. `LOG_LEVEL=DEBUG uv run ...`) overrides the `.env` value.

### 3. Train the model

```bash
uv run python run.py train
```

This loads the data, trains the pipeline, logs the evaluation results and saves the model to `models/pipeline.joblib`.

### 4. Predict with the saved model

```bash
uv run python run.py predict --input examples/sample_input.csv
```

This loads the saved model (no retraining) and predicts each row of the CSV. The expected output for the sample file is:

```
Predicted 3 row(s): ['malignant', 'benign', 'malignant']
```

You can use your own CSV too, as long as it has the same 30 feature columns as the dataset.

### 5. Open the notebook (optional)

Open `notebooks/01_exploration.ipynb` in VS Code and select the `.venv` interpreter as the kernel.

## Error handling

The app handles the failures you'd expect with a clear message and exit code `1`, instead of a crash:

- no saved model yet → "No saved model at …. Run training first."
- the input file doesn't exist
- the input is missing columns, has text instead of numbers, has empty cells, or has no rows

Anything unexpected is not hidden: it shows the full error so it can be fixed.

## Code quality

I use **Ruff** (linting) and **Black** (formatting), both run automatically by **pre-commit** before each commit. To set it up after cloning:

```bash
uv run pre-commit install
```

To run all checks manually:

```bash
uv run pre-commit run --all-files
```