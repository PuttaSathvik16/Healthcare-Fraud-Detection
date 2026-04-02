# Healthcare fraud detection

Production-oriented Python package for ingesting claims data, building features, training a fraud model, and serving scores over HTTP.

## Layout

| Path | Purpose |
|------|---------|
| `src/healthcare_fraud_detection/` | Installable package (`etl`, `preprocessing`, `features`, `models`, `api`, `utils`) |
| `data/raw/` | Drop immutable extracts here |
| `data/processed/` | Cleaned, analysis-ready tables |
| `notebooks/` | Exploratory work (not imported by the package) |
| `models/artifacts/` | Trained `joblib` bundles (gitignored except `.gitkeep`) |

## Quick start

```bash
cd healthcare-fraud-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Environment

Optional `.env` (see `utils.config.Settings` — all variables use prefix `HFD_`):

- `HFD_PROJECT_ROOT` — repo root (defaults to current working directory)
- `HFD_MODEL_REGISTRY_PATH` — path to the trained `model.joblib`
- `HFD_API_HOST`, `HFD_API_PORT` — API bind address
- `HFD_LOG_LEVEL` — logging level

### Train (example)

From Python, with a labeled CSV containing an `is_fraud` column (0/1) and feature columns aligned with your schema:

```python
import pandas as pd
from healthcare_fraud_detection.models.trainer import train_and_save

df = pd.read_parquet("data/processed/claims_clean.parquet")  # or pd.read_csv(...)
train_and_save(df)
```

### Run API

```bash
healthcare-fraud-api
# or
uvicorn healthcare_fraud_detection.api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health` — liveness and whether a model artifact was loaded
- `POST /predict` — JSON body `{"claims": [{...}]}`; extend `ClaimRecord` in `api/schemas.py` to match training columns

Until `models/artifacts/model.joblib` exists, `/predict` returns `503`.

### ETL

```python
from healthcare_fraud_detection.etl import run_etl

run_etl("data/raw/claims.csv")
```

## Development

```bash
pip install -e ".[dev]"
ruff check src
mypy src
pytest
```

## GitHub

Initialize a repository (if you have not already), commit, and push to GitHub:

```bash
cd healthcare-fraud-detection
git init
git add .
git commit -m "Initial commit: healthcare fraud detection pipeline"
git branch -M main
```

Create a **new empty repository** on [GitHub](https://github.com/new) (same name as your folder is fine). Do **not** add a README/license on GitHub if you already committed locally—then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO` with your account and repository name. For SSH, use `git@github.com:YOUR_USERNAME/YOUR_REPO.git` instead.

## Deploy

### Streamlit app (Streamlit Community Cloud)

1. Push the repo to GitHub (steps above).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → pick this repository and branch `main`.
4. **Main file path:** `src/healthcare_fraud_detection/app_streamlit/app.py`
5. **Python version:** 3.10+ (e.g. 3.12) in app settings if offered.
6. **Requirements file:** `requirements.txt` (includes `-e .` so the package imports resolve).

**Secrets:** In the app dashboard → **Secrets**, add at least:

```toml
OPENAI_API_KEY = "sk-..."
```

Optional:

```toml
# If you serve a model from a URL or use custom paths, set via process env in Cloud UI when supported
```

**Model file:** `models/fraud_model.pkl` is **gitignored**. For a hosted demo you must either:

- Run training in a **GitHub Action** and upload the artifact somewhere your app can download at startup, or  
- Temporarily allow the file in git (not ideal for size), or  
- Store the bundle in object storage and fetch it with a small bootstrap script.

Until a model exists at runtime, the Streamlit UI will show “Scoring engine unavailable.”

### FastAPI (example platforms)

Build image or use a PaaS that runs:

```bash
pip install -r requirements.txt
uvicorn healthcare_fraud_detection.api.main:app --host 0.0.0.0 --port $PORT
```

Set `PORT` from the platform. Configure `OPENAI_API_KEY` for `/explain`. Same model artifact caveat as above.

## Notes

- Replace placeholder columns (`billed_amount`, `claim_count`, etc.) with your real schema consistently across ETL, features, and API schemas.
- For regulated environments, add audit logging, authn/z on the API, and separate training/serving configs before production deploy.
