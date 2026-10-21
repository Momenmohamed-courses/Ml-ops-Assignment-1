# Breast Cancer Classifier: From Notebook to a Validated ML API

This project is my work for the Production AI Engineering course, built over two assignments on the same codebase:

- **Assignment 01:** take a simple ML experiment from a notebook and turn it into a clean, reproducible Python project (config, logging, training, saving, prediction from the command line).
- **Assignment 02:** put that same saved model behind a **FastAPI** web API, with **Pydantic** checking every request before it reaches the model.

The goal in both was good engineering, not the highest possible accuracy.

## What it does

It predicts whether a breast tumor is **malignant** (cancerous) or **benign**, from 30 measurements taken from a scan of the cell nuclei. The model is trained once and saved to a file. After that, predictions (from the command line or from the API) always load that saved file. Nothing is retrained at prediction time.

## Dataset

I used the **Breast Cancer Wisconsin** dataset, which ships with scikit-learn.

- 569 patients, 30 numeric features, no missing values
- Labels: `0 = malignant`, `1 = benign`
- Slightly unbalanced: 357 benign (~63%) vs 212 malignant (~37%)

Because of that imbalance, I split the data in a stratified way and I don't rely on accuracy alone.

## The ML pipeline

```
Load data → Inspect → Split train/test → Scale → Fit model → Evaluate → Save → Reload → Predict
```

- **Split:** 80% train / 20% test, stratified, with a fixed random seed so results are the same on every run
- **Preprocessing:** `StandardScaler`. The features live on very different scales (area is in the hundreds, smoothness is around 0.1), and Logistic Regression learns better when they're comparable
- **No data leakage:** the scaler and the model sit together in one scikit-learn `Pipeline`, so the scaler only ever learns from the training data. That same fitted pipeline is what gets saved and reused for every prediction

## Model choice

**Logistic Regression.** It's simple, fast, easy to explain, and a solid baseline for a small numeric dataset like this one. It also gives a probability with each prediction, which the API returns. No tuning was needed, so I kept the defaults apart from `max_iter=1000`.

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

I report precision/recall for the malignant class because that's the mistake that matters most here: missing a real cancer is much worse than a false alarm. The model missed 1 of 42 malignant cases.

**Something I learned along the way:** my first notebook run had no random seed in the split, so the score kept jumping between 0.96 and 0.98 every time I re-ran it. Fixing the seed made the results reproducible and identical between the notebook and the Python code.

## Project structure

```
Ml-ops-Assignment-1/
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
│   ├── main.py                # the training and prediction workflows
│   ├── schemas.py             # NEW: Pydantic request/response schemas for the API
│   └── api.py                 # NEW: FastAPI app with /health and /predict
├── examples/
│   ├── sample_input.csv       # 3 test patients, for CLI prediction
│   └── sample_request.json    # NEW: one patient as JSON, for the API
├── models/                    # the trained model is saved here (not committed)
├── run.py                     # command-line entry point: train or predict
├── .env.example               # the settings you can change
├── .pre-commit-config.yaml    # runs Ruff and Black before every commit
├── pyproject.toml / uv.lock   # dependencies, pinned for reproducibility
└── README.md
```

The notebook is my experimentation record. The `production_app` package is the source of truth for the actual application.

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

`uv sync` creates the `.venv` and installs the exact versions from `uv.lock` (including FastAPI, Pydantic and uvicorn).

### 2. Configure

```bash
cp .env.example .env
```

The defaults work out of the box. These are the settings you can change in `.env`:

| Variable | Default | What it does |
|---|---|---|
| `MODEL_PATH` | `models/pipeline.joblib` | where the trained model is saved and loaded from (used by the CLI and the API) |
| `LOG_LEVEL` | `INFO` | how much logging to show (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `APP_ENV` | `development` | which environment the app runs in |
| `TEST_SIZE` | `0.2` | share of data kept for testing (must be between 0 and 1) |
| `RANDOM_SEED` | `42` | keeps the split the same on every run |
| `MAX_ITER` | `1000` | max training iterations for Logistic Regression |

If `.env` is missing, the app falls back to these defaults. A setting given in the terminal (e.g. `LOG_LEVEL=DEBUG uv run ...`) overrides the `.env` value.

### 3. Train the model (creates the saved artifact)

The trained model is not stored in the repository, so you create it once after cloning:

```bash
uv run python run.py train
```

This loads the data, trains the pipeline, logs the evaluation results and saves the model to `models/pipeline.joblib`. Both the CLI and the API use this file.

### 4. Predict from the command line

```bash
uv run python run.py predict --input examples/sample_input.csv
```

This loads the saved model and predicts each row of the CSV. Expected output for the sample file:

```
Predicted 3 row(s): ['malignant', 'benign', 'malignant']
```

You can use your own CSV too, as long as it has the same 30 feature columns as the dataset.

## The prediction API (Assignment 02)

### What Assignment 02 adds

- `production_app/schemas.py`: Pydantic schemas describing a valid request (one patient) and the responses
- `production_app/api.py`: the FastAPI app with `GET /health` and `POST /predict`
- `production_app/model_service.py`: one new function, `predict_one`, which predicts a single patient and also returns the probability. Everything else in it is unchanged
- `examples/sample_request.json`: a ready-made request body
- new dependencies: `fastapi`, `uvicorn` (Pydantic comes with FastAPI)

Nothing from Assignment 01 changed behaviour: training, the CLI and the saved model work exactly as before.

### How a request flows

```
HTTP/JSON request → FastAPI endpoint → Pydantic validation → saved scikit-learn pipeline → JSON response
```

1. When the API starts, it loads the saved model **once** and keeps it in memory. If the file is missing, the API still starts, so `/health` can report the problem.
2. A client sends one patient as JSON to `/predict`.
3. **Pydantic** checks the JSON against the schema: all 30 fields present, all numbers, none negative, no unknown fields. If anything is wrong, the client gets a `422` right away and the model is never called.
4. Valid data goes to `predict_one` in `model_service.py`, which runs it through the saved pipeline (the same fitted scaler + model).
5. The answer comes back as JSON: the class, a readable label and the probability.

The endpoint functions stay small on purpose: all the ML logic lives in `model_service.py`, shared with the CLI.

### Start and stop the API

Make sure you've trained the model first (step 3), then:

```bash
uv run uvicorn production_app.api:app
```

The API runs at `http://127.0.0.1:8000`. To stop it, press **Ctrl + C** in that terminal.

Open `http://127.0.0.1:8000/docs` in a browser for the interactive documentation (Swagger UI), where you can see every endpoint and schema and try requests.

### `GET /health`

Tells you whether the service is up **and** able to predict.

```bash
curl http://127.0.0.1:8000/health
```

| Situation | Status | Response |
|---|---|---|
| model loaded | `200` | `{"status": "ok", "model_loaded": true}` |
| model file missing | `503` | `{"status": "unavailable", "model_loaded": false}` |

### `POST /predict`

Send one patient, get the prediction.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

Request body (shortened, the full example is in `examples/sample_request.json`):

```json
{
  "mean radius": 19.55,
  "mean texture": 28.77,
  "mean perimeter": 133.6,
  "mean area": 1207.0,
  "...": "all 30 features"
}
```

Successful response (`200`):

```json
{"predicted_class": 0, "label": "malignant", "probability": 0.9999999411175814}
```

- `predicted_class`: `0` or `1`
- `label`: `"malignant"` or `"benign"`
- `probability`: how confident the model is in that class, between 0 and 1

### The feature schema

Every request needs **all 30 features**, using the exact names from the dataset (with spaces). They come from 10 measurements, each in 3 versions:

| Measurement | Average | Variation | Largest |
|---|---|---|---|
| radius | `mean radius` | `radius error` | `worst radius` |
| texture | `mean texture` | `texture error` | `worst texture` |
| perimeter | `mean perimeter` | `perimeter error` | `worst perimeter` |
| area | `mean area` | `area error` | `worst area` |
| smoothness | `mean smoothness` | `smoothness error` | `worst smoothness` |
| compactness | `mean compactness` | `compactness error` | `worst compactness` |
| concavity | `mean concavity` | `concavity error` | `worst concavity` |
| concave points | `mean concave points` | `concave points error` | `worst concave points` |
| symmetry | `mean symmetry` | `symmetry error` | `worst symmetry` |
| fractal dimension | `mean fractal dimension` | `fractal dimension error` | `worst fractal dimension` |

Rules for every value:

- must be a **number** (`12` and `12.5` are fine, the text `"12.5"` is not)
- must be **0 or more** (not "more than 0", because some features really are 0 in the data, e.g. `mean concavity`)
- `NaN` and `Infinity` are not allowed
- **unknown fields are rejected**, so a typo like `"mean radios"` is reported instead of silently ignored

### Validation errors

Bad requests get a `422` with a clear explanation. Each error says **what** went wrong (`type`), **where** (`loc`) and a readable message (`msg`). Some examples I checked:

| What I sent | `type` | `msg` |
|---|---|---|
| `mean radius` left out | `missing` | Field required |
| `"mean radius": "12.5"` | `float_type` | Input should be a valid number |
| `"mean radius": -1` | `greater_than_equal` | Input should be greater than or equal to 0 |
| extra field `"mean radios"` | `extra_forbidden` | Extra inputs are not permitted |
| broken JSON | `json_invalid` | JSON decode error |

For example, leaving out `mean radius` returns:

```json
{"detail": [{"type": "missing", "loc": ["body", "mean radius"], "msg": "Field required", "input": {"...": "..."}}]}
```

If the model file can't be loaded, `/predict` returns `503` with a safe message:

```json
{"detail": "Model is not available. Please try again later."}
```

The client never sees stack traces, file paths or settings. Those details only go to the server's log, where they're useful for debugging. Any unexpected error returns a plain `500 Internal Server Error`.

### Troubleshooting

- **`/health` says `unavailable`, or `/predict` returns `503`:** there's no saved model at `MODEL_PATH`. Run `uv run python run.py train`, then restart the API (it loads the model only at startup).
- **`address already in use` when starting the API:** another process is still using port 8000, often an API you started earlier. Find it with `lsof -i :8000` and stop it with `kill <PID>`, then start again.
- **`{"detail":"Not Found"}`:** you opened an address that doesn't exist, like `http://127.0.0.1:8000/`. Use `/health`, `/predict` or `/docs`.
- **`422` but the values look right:** check the field names match exactly (spaces, no typos) and that numbers aren't sent as text in quotes.
- **`zsh: command not found: uv`:** open a new terminal or run `source $HOME/.local/bin/env` after installing uv.

## Error handling in the CLI

`run.py` handles the expected failures with a clear message and exit code `1`, instead of a crash:

- no saved model yet → "No saved model at …. Run training first."
- the input file doesn't exist
- the input is missing columns, has text instead of numbers, has empty cells, or has no rows

Anything unexpected isn't hidden: it shows the full error so it can be fixed.

## Code quality

I use **Ruff** (linting) and **Black** (formatting, including the notebook), both run automatically by **pre-commit** before each commit. To set it up after cloning:

```bash
uv run pre-commit install
```

To run all checks manually:

```bash
uv run pre-commit run --all-files
```