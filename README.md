# Breast Cancer Classifier: From Notebook to a Production-Ready ML App

This project is my work for the Production AI Engineering course, built across three assignments on one codebase:

- **Assignment 01:** turn a notebook experiment into a clean, reproducible Python project (config, logging, training, saving, CLI prediction).
- **Assignment 02:** serve the saved model through a **FastAPI** API, with **Pydantic** validating every request.
- **Assignment 03:** reduce the inputs with **feature selection**, add a **Streamlit** interface, package the API with **Docker**, and cover it with **pytest**.

The goal throughout was good engineering and reproducibility, not the highest possible accuracy.

## What it does

It predicts whether a breast tumor is **malignant** (cancerous) or **benign** from measurements taken from a scan of the cell nuclei. The model is trained once and saved to a file. Every prediction (CLI, API, Streamlit) loads that saved file; nothing is retrained at prediction time.

## How it fits together

```
TRAIN:  dataset → split → select features (training data only) → fit scaler + model → evaluate → save model + feature list
SERVE:  Streamlit form → HTTP → FastAPI + Pydantic → saved model → JSON → Streamlit result
RUN:    Docker image → running FastAPI container (model mounted) → /health and /predict
```

## Dataset

**Breast Cancer Wisconsin**, shipped with scikit-learn.

- 569 patients, 30 numeric features, no missing values
- Labels: `0 = malignant`, `1 = benign`
- Slightly unbalanced: 357 benign (~63%) vs 212 malignant (~37%), so I split in a stratified way and don't rely on accuracy alone

## Model

A scikit-learn `Pipeline` of **`StandardScaler` + `LogisticRegression`** (`max_iter=1000`), trained on the **11 selected features**. Logistic Regression is simple, fast, easy to explain, and gives a probability with each prediction.

The scaler and the model are fitted together on the training data only, and saved together, so predictions always use exactly the same preprocessing as training.

## Feature selection (Assignment 03)

### The three families I researched

| Family | How it works | Example | Strength | Limitation |
|---|---|---|---|---|
| Filter | Scores each feature on its own with a statistic, keeps the top ones | SelectKBest (ANOVA F-test) | Very fast, model-independent | Ignores redundancy: keeps several features that say the same thing |
| Wrapper | Trains the model on feature subsets and keeps what performs best | RFE | Picks features that work well together for this model | Slow: trains the model many times |
| Embedded | Selection happens during training: L1 pushes weak weights to exactly 0 | L1 Logistic Regression + SelectFromModel | One training run, model-aware, handles redundancy | The number of kept features depends on the strength C, not chosen directly |

I also considered **PCA**, but it creates new mixed components from all 30 inputs, so users would still need to enter 30 values. It reduces dimensions, but it doesn't select features.

### What I chose and why

**Embedded, with L1 Logistic Regression** (`l1_ratio=1.0`, `solver="liblinear"`). L1 handles the many redundant features in this dataset (radius, perimeter and area all describe size), and it does the selection in a single training run.

I chose the strength **C** with 5-fold stratified cross-validation **on the training set only**, with the selector inside the pipeline, so no information from the test set (or from the validation fold) influences which features are kept:

| C | Features kept | CV accuracy | CV recall (malignant) |
|---|---|---|---|
| baseline | 30 | 0.978 | 0.965 |
| 1.0 | 16 | 0.963 | 0.924 |
| 0.5 | 12 | 0.965 | 0.935 |
| **0.3** | **11** | **0.974** | **0.953** |
| 0.2 | 10 | 0.967 | 0.947 |
| 0.1 | 8 | 0.965 | 0.941 |
| 0.05 | 7 | 0.965 | 0.947 |

**C = 0.3** is the best balance: 30 → 11 inputs with the smallest loss in recall for malignant, which is the costliest mistake here (a missed cancer). Interesting detail: keeping more features (16) gave the worst recall, so more inputs isn't automatically better.

### The 11 selected features (in the order the model expects)

`mean texture`, `mean concave points`, `radius error`, `fractal dimension error`, `worst radius`, `worst texture`, `worst area`, `worst smoothness`, `worst concavity`, `worst concave points`, `worst symmetry`

Most are "worst" measurements (the most abnormal cell in the sample), and only one or two of each redundant group survived.

### Baseline vs selected, on the test set (114 patients, used once)

| Model | Features | Accuracy | Precision (malignant) | Recall (malignant) | F1 (malignant) |
|---|---|---|---|---|---|
| Baseline | 30 | 0.982 | 0.976 | 0.976 | 0.976 |
| **Selected** | **11** | **0.982** | **0.976** | **0.976** | **0.976** |

Confusion matrix of the selected model (rows = real, columns = predicted): `[[41, 1], [1, 71]]`, so 1 of 42 malignant cases missed and 1 false alarm.

Honest reading: the test set is small, so "identical on test" and "slightly lower in cross-validation" are both true. Together they say 19 features can go with little or no loss. For a real medical product I'd watch recall closely and might keep a few more features.

The full experiment is in `notebooks/02_feature_selection.ipynb`. Every training run also logs the baseline and the selected model side by side.

## Project structure

```
Ml-ops-Assignment-1/
├── notebooks/
│   ├── 01_exploration.ipynb       # Task 01: data checks and the baseline model
│   └── 02_feature_selection.ipynb # Task 03: research + L1 experiment
├── production_app/
│   ├── config.py                  # settings from .env, with safe defaults
│   ├── logging_config.py          # logging set up once
│   ├── data.py                    # load, split, read prediction CSVs
│   ├── preprocessing.py           # the (unfitted) scaler
│   ├── feature_selection.py       # L1 selection on training data only
│   ├── training.py                # builds and fits the pipeline
│   ├── evaluation.py              # metrics, labelled per model
│   ├── model_service.py           # save/load model + feature list, validate, predict
│   ├── main.py                    # training and prediction workflows
│   ├── schemas.py                 # Pydantic request/response schemas (11 features)
│   └── api.py                     # FastAPI: /health and /predict
├── tests/
│   ├── conftest.py                # trains a test model once, in a temp folder
│   ├── test_model.py              # model workflow tests
│   └── test_api.py                # API success and error tests
├── examples/
│   ├── sample_input.csv           # 3 test patients (11 features), for the CLI
│   └── sample_request.json        # 1 patient (11 features), for the API
├── models/                        # trained artifacts (created by training, not committed)
├── streamlit_app.py               # Streamlit UI, calls the API over HTTP
├── run.py                         # CLI: train or predict
├── Dockerfile / .dockerignore     # API container
├── .env.example                   # all settings, with safe example values
└── pyproject.toml / uv.lock       # dependencies, pinned
```

## Setup

### Prerequisites

- macOS or Linux, Git
- [uv](https://docs.astral.sh/uv/) (installs the right Python for you)
- [Docker](https://www.docker.com/) (only for the container part)

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal, or run `source $HOME/.local/bin/env`.

### 1. Clone and install

```bash
git clone https://github.com/Momenmohamed-courses/Ml-ops-Assignment-1.git
cd Ml-ops-Assignment-1
uv sync
```

`uv sync` installs the exact versions from `uv.lock`: the app, plus the `dev` tools (pytest, Ruff, Black, pre-commit) and the `ui` group (Streamlit).

### 2. Configure

```bash
cp .env.example .env
```

The defaults work as they are:

| Variable | Default | What it does |
|---|---|---|
| `MODEL_PATH` | `models/pipeline.joblib` | where the model is saved/loaded; the feature list sits next to it as `pipeline.features.json` |
| `LOG_LEVEL` | `INFO` | logging detail (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `APP_ENV` | `development` | environment name |
| `TEST_SIZE` | `0.2` | share of data kept for testing (between 0 and 1) |
| `RANDOM_SEED` | `42` | makes the split reproducible |
| `MAX_ITER` | `1000` | max training iterations for Logistic Regression |
| `SELECTION_C` | `0.3` | L1 feature-selection strength (smaller = fewer features; must be > 0) |
| `API_URL` | `http://127.0.0.1:8000` | the API address Streamlit calls |
| `REQUEST_TIMEOUT` | `5` | seconds Streamlit waits for the API |

If `.env` is missing, these defaults are used. A variable set in the terminal (e.g. `API_URL=http://127.0.0.1:8001 uv run ...`) overrides `.env`.

### 3. Train (recreates the model artifacts)

The trained model is not committed, so this is the first thing to run after cloning:

```bash
uv run python run.py train
```

It logs the baseline (30 features) and the selected model (11 features) side by side, and writes two files:

- `models/pipeline.joblib`: the fitted scaler + model
- `models/pipeline.features.json`: the 11 selected feature names, in order

### 4. Predict from the command line

```bash
uv run python run.py predict --input examples/sample_input.csv
```

Expected: `Predicted 3 row(s): ['malignant', 'benign', 'malignant']`. A CSV needs the 11 selected columns; a missing one is rejected with a clear error.

## The API

### Start and stop it locally

```bash
uv run uvicorn production_app.api:app
```

It runs at `http://127.0.0.1:8000`; interactive docs at `http://127.0.0.1:8000/docs`. Stop it with **Ctrl + C**. The model is loaded once at startup.

### Request flow

1. The client sends one patient as JSON to `POST /predict`.
2. Pydantic checks it: exactly the 11 selected fields, all numbers, none negative, no NaN/Infinity, no unknown fields. Anything wrong → `422`, and the model is never called.
3. Valid data goes to `predict_one` in `model_service.py`, which runs the saved pipeline.
4. The API answers with the class, a readable label and the probability.

### `GET /health`

| Situation | Status | Response |
|---|---|---|
| model loaded | `200` | `{"status": "ok", "model_loaded": true}` |
| model missing | `503` | `{"status": "unavailable", "model_loaded": false}` |

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

Request body: all 11 selected features (full example in `examples/sample_request.json`):

```json
{
  "mean texture": 28.77,
  "mean concave points": 0.1144,
  "radius error": 0.8426,
  "...": "the other 8 selected features"
}
```

Response (`200`):

```json
{"predicted_class": 0, "label": "malignant", "probability": 0.9999998653482068}
```

### Errors

| Situation | Status | Example |
|---|---|---|
| a selected feature missing | `422` | `"type": "missing"`, `"msg": "Field required"` |
| a removed feature sent (e.g. `mean radius`) | `422` | `"type": "extra_forbidden"` |
| text instead of a number | `422` | `"type": "float_type"` |
| a negative value | `422` | `"type": "greater_than_equal"` |
| broken JSON | `422` | `"type": "json_invalid"` |
| model not available | `503` | `{"detail": "Model is not available. Please try again later."}` |

Clients never see stack traces, file paths or settings; those only go to the server log.

## Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

It opens at `http://localhost:8501`. Fill in the 11 values (or click **Fill sample values**) and click **Predict**. The page only talks to the API over HTTP; it never loads the model.

It shows clear messages for: an empty field (nothing is sent), values the API rejects (with the reason per field), the API not running, a timeout, and an unavailable model.

The form's fields come from the API's request schema, so the UI, the API and the saved model always agree on the 11 inputs.

## Docker (the API)

The image contains only the API: Python 3.13 slim, the main dependencies from `uv.lock` (no dev tools, no Streamlit), and `production_app/`. It runs as a non-root user.

**How the model gets in:** it's **not** baked into the image. You train once (step 3), then **mount** the `models/` folder read-only when starting the container. Retraining doesn't need a rebuild. Without the mount, the API still starts and `/health` reports `503`.

```bash
# build
docker build -t breast-cancer-api .

# run (container port 8000 → your port 8001, model folder mounted read-only)
docker run -d --name breast-cancer-api -p 8001:8000 \
  -v "$(pwd)/models:/app/models:ro" breast-cancer-api

# check
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" -d @examples/sample_request.json

# logs
docker logs breast-cancer-api

# stop and remove
docker stop breast-cancer-api
docker rm breast-cancer-api
```

**Connect Streamlit to the container** by pointing `API_URL` at the mapped port:

```bash
API_URL=http://127.0.0.1:8001 uv run streamlit run streamlit_app.py
```

## Tests

```bash
uv run pytest -v
```

12 tests, no running server or local model needed (they train their own model once, in a temporary folder):

- **Model:** selection keeps fewer, real features; the saved model loads and predicts with the selected feature shape without retraining; a missing model raises a clear error; the feature list saves and loads identically; **the API schema matches the saved model's features**
- **API:** `/health` 200 with a model; valid `/predict` 200 with class/label/probability; missing, extra (removed) and text fields → 422; broken JSON → 422; no model → `/health` and `/predict` 503

## Code quality

Ruff (linting) and Black (formatting, including notebooks) run automatically before every commit through pre-commit:

```bash
uv run pre-commit install          # once after cloning
uv run pre-commit run --all-files  # run all checks manually
```

## Troubleshooting

- **`/health` says `unavailable` or `/predict` returns 503:** no model at `MODEL_PATH` (or, in Docker, the `models/` folder isn't mounted). Run `uv run python run.py train`, then restart the API or container.
- **`address already in use`:** something already uses that port. Find it with `lsof -i :8000` and stop it with `kill <PID>`, or map another port.
- **Streamlit says it can't reach the API:** the API isn't running, or `API_URL` points to the wrong port (e.g. 8001 for the container).
- **422 although values look fine:** check the field names (exact, with spaces), that you're sending only the 11 selected features, and that decimals use a dot (`0.5`, not `0,5`).
- **`zsh: command not found: uv`:** open a new terminal or run `source $HOME/.local/bin/env`.

## Limitations

- One small public dataset (569 patients); results on real clinical data could differ.
- The test set is small (114 patients), so small differences between models are within noise; cross-validation gives the more reliable picture.
- This is a learning project, not a medical tool.