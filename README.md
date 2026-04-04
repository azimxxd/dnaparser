# DNAParser

FastAPI service for VCF variant risk analysis.

The API accepts a `.vcf` / `.vcf.gz` file, scores variants with a trained XGBoost model, and returns top high-risk mutations with optional disease mapping.

## 1. Run Locally

### Prerequisites
- Python 3.12+
- `pip`

### Setup and start API
```bash
cd /home/azamat/projects/dnaparser
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Test with sample file
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample.vcf"
```

## 2. Run with Docker Compose

```bash
cd /home/azamat/projects/dnaparser
docker compose up --build
```

Service will be available at `http://localhost:8000`.

Quick test:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample.vcf"
```

## 3. Retrain the Model

### Dataset requirement
`train_model.py` expects `clinvar.vcf.gz` in project root:

`/home/azamat/projects/dnaparser/clinvar.vcf.gz`

Note: this file is not tracked in GitHub (size > 100MB).

### Train
```bash
cd /home/azamat/projects/dnaparser
source .venv/bin/activate
python train_model.py
```

### Expected output artifacts
- `model.joblib` — trained XGBoost model
- `encoders.joblib` — encoder payload (`schema_version`, `feature_cols`, `encoders`)
- `disease_map.joblib` — `CHROM_POS_REF_ALT -> disease`

After retraining, restart the API so it loads the new artifacts.

## 4. API Example (`/analyze`)

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample.vcf"
```

Typical response shape:
```json
{
  "status": "completed",
  "total_variants_scanned": 200000,
  "top_risks": [
    {
      "chromosome": "chr17",
      "position": "43045710",
      "mutation": "A -> G",
      "risk_score": 0.942301,
      "associated_disease": "..."
    }
  ]
}
```
