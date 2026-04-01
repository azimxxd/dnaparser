"""
Training script for the VCF variant pathogenicity classifier.

Parses ClinVar VCF, extracts labels and INFO-derived features,
trains an XGBoost classifier with train/test evaluation, and exports
model artifacts to disk.

Usage:
    python train_model.py

Outputs:
    model.joblib       — trained XGBClassifier
    encoders.joblib    — dict payload with LabelEncoders + feature ordering
    disease_map.joblib — CHROM_POS_REF_ALT -> disease name
"""

from __future__ import annotations

import gzip
import io
import re

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

CLINVAR_PATH = "clinvar.vcf.gz"

PATHOGENIC_LABELS = {
    "Pathogenic",
    "Likely_pathogenic",
    "Pathogenic/Likely_pathogenic",
}
BENIGN_LABELS = {
    "Benign",
    "Likely_benign",
    "Benign/Likely_benign",
}


def _parse_info(info: str) -> dict[str, str]:
    """Parse a semicolon-delimited VCF INFO field into key-value pairs."""
    parsed: dict[str, str] = {}
    for item in info.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def _parse_first_float(raw_value: str | None) -> float | None:
    """Parse first float from values like '0.12,0.03'; return None on failure."""
    if not raw_value:
        return None
    first = raw_value.split(",", 1)[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def _normalize_gene_name(raw_value: str | None) -> str:
    """Extract a stable gene symbol from ClinVar GENEINFO."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_GENE"
    # Example: BRCA1:672|NBR2:10230 -> BRCA1
    first_gene = raw_value.split("|", 1)[0]
    symbol = first_gene.split(":", 1)[0].strip()
    return symbol or "UNKNOWN_GENE"


def map_label(clnsig: str) -> int | None:
    """Map CLNSIG to binary target; return None for ambiguous labels."""
    # CLNSIG may contain compound labels (e.g., 'Pathogenic|risk_factor').
    tokens = [t.strip() for t in re.split(r"[|,/]", clnsig) if t.strip()]

    if any(token in PATHOGENIC_LABELS for token in tokens) or clnsig in PATHOGENIC_LABELS:
        return 1
    if any(token in BENIGN_LABELS for token in tokens) or clnsig in BENIGN_LABELS:
        return 0
    return None


# ============================================================================
# 1. Parse ClinVar VCF — extract features + labels
# ============================================================================

print(f"Parsing {CLINVAR_PATH}...")

rows: list[tuple[str, int, str, str, float | None, float | None, str, str, str, str]] = []

with io.TextIOWrapper(gzip.open(CLINVAR_PATH, "rb"), encoding="utf-8") as fh:
    for line in fh:
        if line[0] == "#":
            continue

        fields = line.split("\t", 8)
        chrom = fields[0]
        pos = fields[1]
        ref = fields[3]
        alt = fields[4]
        info = fields[7]

        info_map = _parse_info(info)

        clnsig = info_map.get("CLNSIG")
        if not clnsig:
            continue

        disease = info_map.get("CLNDN", "")
        af_esp = _parse_first_float(info_map.get("AF_ESP"))
        af_exac = _parse_first_float(info_map.get("AF_EXAC"))

        clnvc = info_map.get("CLNVC", "UNKNOWN_CLNVC") or "UNKNOWN_CLNVC"
        if clnvc == ".":
            clnvc = "UNKNOWN_CLNVC"

        gene = _normalize_gene_name(info_map.get("GENEINFO"))

        rows.append((chrom, int(pos), ref, alt, af_esp, af_exac, clnvc, gene, clnsig, disease))

print(f"Parsed {len(rows)} variants with CLNSIG labels")

feature_df = pd.DataFrame(
    rows,
    columns=[
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "AF_ESP",
        "AF_EXAC",
        "CLNVC",
        "GENEINFO",
        "CLNSIG",
        "CLNDN",
    ],
)

# ============================================================================
# 2. Clean labels and filter dataset
# ============================================================================

feature_df["label"] = feature_df["CLNSIG"].map(map_label)
feature_df = feature_df.dropna(subset=["label"]).copy()
feature_df["label"] = feature_df["label"].astype(int)

print(f"After filtering to Pathogenic/Benign: {len(feature_df)} rows")
print(f"Class distribution:\n{feature_df['label'].value_counts().to_string()}\n")

# Keep only SNVs for this model design.
snv_mask = (feature_df["REF"].str.len() == 1) & (feature_df["ALT"].str.len() == 1)
feature_df = feature_df[snv_mask].copy()
print(f"After filtering to SNVs only: {len(feature_df)} rows")

# Normalize categorical INFO-derived features.
feature_df["CLNVC"] = (
    feature_df["CLNVC"].fillna("UNKNOWN_CLNVC").replace("", "UNKNOWN_CLNVC")
)
feature_df["GENEINFO"] = (
    feature_df["GENEINFO"].fillna("UNKNOWN_GENE").replace("", "UNKNOWN_GENE")
)

# AF features remain numeric with NaN for missing values.
feature_df["AF_ESP"] = pd.to_numeric(feature_df["AF_ESP"], errors="coerce")
feature_df["AF_EXAC"] = pd.to_numeric(feature_df["AF_EXAC"], errors="coerce")

# ============================================================================
# 3. Encode categorical columns
# ============================================================================

categorical_cols = ("CHROM", "REF", "ALT", "CLNVC", "GENEINFO")
encoders: dict[str, LabelEncoder] = {}

for col in categorical_cols:
    le = LabelEncoder()
    feature_df[col] = le.fit_transform(feature_df[col])
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# ============================================================================
# 4. Train/test split and XGBoost training
# ============================================================================

feature_cols = ["CHROM", "POS", "REF", "ALT", "AF_ESP", "AF_EXAC", "CLNVC", "GENEINFO"]
X = feature_df[feature_cols].astype("float32").values
y = feature_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(
    "Class balance — "
    f"Benign: {n_neg}, Pathogenic: {n_pos}, scale_pos_weight: {scale_pos_weight:.2f}"
)

print("\nTraining XGBClassifier...")

model = XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ============================================================================
# 5. Evaluate
# ============================================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Test Set Evaluation ---")
print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ============================================================================
# 6. Build disease lookup map from pathogenic variants
# ============================================================================

print("\nBuilding disease_map from pathogenic variants...")

pathogenic_df = feature_df[feature_df["label"] == 1].copy()
for col in ("CHROM", "REF", "ALT"):
    pathogenic_df[col] = encoders[col].inverse_transform(pathogenic_df[col])

disease_map: dict[str, str] = {}
for _, row in pathogenic_df.iterrows():
    key = f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}"
    disease = row["CLNDN"]

    if disease and disease != "not_provided":
        disease = disease.replace("_", " ").replace("|", ", ")
        disease_map[key] = disease

print(f"Disease map: {len(disease_map)} pathogenic variants with known diseases")

# ============================================================================
# 7. Export artifacts
# ============================================================================

joblib.dump(model, "model.joblib")
joblib.dump(
    {
        "schema_version": 2,
        "feature_cols": feature_cols,
        "encoders": encoders,
    },
    "encoders.joblib",
)
joblib.dump(disease_map, "disease_map.joblib")

print("\nSaved: model.joblib, encoders.joblib, disease_map.joblib")
print("Done. Restart the FastAPI server to use the new model.")
