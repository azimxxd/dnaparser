# DNA Parser

VCF analysis backend with a compact web app.

The project:
- accepts `.vcf` and `.vcf.gz`
- scores variants with an `XGBoost` model
- serves a web UI built with plain `HTML`, `CSS`, and `JavaScript`
- can return both analysis output and a saved report JSON

## What The Model Does

The model does **not** predict the exact chance that a person has a disease.
It estimates how similar a variant is to known pathogenic variants from the training data.

Current scope:
- trained on `ClinVar`
- focused on `SNV`
- uses features like `CHROM`, `POS`, `REF`, `ALT`, `AF_ESP`, `AF_EXAC`, `AF_TGP`, `CLNVC`, `GENEINFO`, `MC`, `ORIGIN`, `IS_TRANSITION`

Important:
- this is a prioritization tool, not a diagnosis engine
- broader variant classes need separate validation

## Run On Windows

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Start the server:

```powershell
.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the app in the browser:

```text
http://127.0.0.1:8000/
```

If you change `main.py`, restart the server.

## What You Can Do In The App

- upload your own `VCF`
- switch between `Analyze` and `Report`
- run built-in demo cases
- see a compact on-page preview
- download the current JSON result to a file
- download a saved server-side report JSON

## Quick Local Check Without The Browser

```powershell
.venv\Scripts\python benchmark_demo.py
```

This runs:
- `high_signal.vcf`
- `low_signal.vcf`
- `mixed_signal.vcf`

## Useful API Endpoints

Health and model info:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/ready
curl.exe http://127.0.0.1:8000/model-info
curl.exe http://127.0.0.1:8000/demo-cases
```

Analyze a file:

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze -F "file=@demo_cases/high_signal.vcf"
```

Generate a saved report:

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze/report -F "file=@demo_cases/high_signal.vcf"
```

Read or download a saved report:

```powershell
curl.exe http://127.0.0.1:8000/reports/REPORT_ID
curl.exe http://127.0.0.1:8000/reports/REPORT_ID/download
```

Saved reports are written to `saved_reports/`.

## Tests

Run tests:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Current tests cover:
- `/`
- `/health`
- `/ready`
- `/model-info`
- `/demo-cases`
- `/analyze`
- `/analyze/report`
- `/reports/{report_id}`
- `/reports/{report_id}/download`

## Retrain The Model

If you want to retrain:

```powershell
.venv\Scripts\python train_model.py
```

Then refresh demo cases:

```powershell
.venv\Scripts\python refresh_demo_cases.py
```

Then restart the API.

## Main Files

- `main.py` - FastAPI app and analysis logic
- `frontend/index.html` - web app markup
- `frontend/assets/styles.css` - web app styles
- `frontend/assets/app.js` - web app logic
- `train_model.py` - model training
- `benchmark_demo.py` - local demo runner
- `refresh_demo_cases.py` - demo case refresh
- `vcf_parser.py` - streaming VCF parser
- `tests/test_api.py` - API tests

## Limits

- current model is designed for `SNV`
- output does not replace clinician review
- specialist hints and follow-up guidance are heuristic
