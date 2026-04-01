"""Training script for the VCF variant pathogenicity classifier.

Parses ClinVar VCF, extracts safe INFO-derived features, trains an
XGBoost classifier, evaluates it on validation and holdout test splits,
and exports model artifacts to disk.

Usage:
    python train_model.py
    python train_model.py --sample-limit 200000

Outputs:
    model.joblib         - trained XGBClassifier
    encoders.joblib      - dict payload with LabelEncoders + feature ordering
    disease_map.joblib   - CHROM_POS_REF_ALT -> disease name
    model_metrics.json   - training/evaluation summary
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DEFAULT_CLINVAR_PATH = "clinvar.vcf.gz"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_RANDOM_STATE = 42

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
TRANSITION_PAIRS = {"AG", "GA", "CT", "TC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ClinVar pathogenicity model from a VCF file.",
    )
    parser.add_argument(
        "--clinvar-path",
        default=DEFAULT_CLINVAR_PATH,
        help="Path to clinvar.vcf.gz",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model artifacts will be saved.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on parsed rows for smoke tests and quick experiments.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for data splits and model training.",
    )
    return parser.parse_args()


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
    if not raw_value or raw_value == ".":
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
    first_gene = raw_value.split("|", 1)[0]
    symbol = first_gene.split(":", 1)[0].strip()
    return symbol or "UNKNOWN_GENE"


def _normalize_clnvc(raw_value: str | None) -> str:
    """Normalize ClinVar variant class."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_CLNVC"
    return raw_value.strip() or "UNKNOWN_CLNVC"


def _normalize_mc(raw_value: str | None) -> str:
    """Normalize molecular consequence into a single stable category."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_MC"
    first = raw_value.split(",", 1)[0]
    if "|" in first:
        consequence = first.split("|", 1)[1].strip()
        return consequence or "UNKNOWN_MC"
    return first.strip() or "UNKNOWN_MC"


def _normalize_origin(raw_value: str | None) -> str:
    """Normalize origin field into a stable single token."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_ORIGIN"
    return raw_value.split(",", 1)[0].strip() or "UNKNOWN_ORIGIN"


def _is_transition(ref: str, alt: str) -> int:
    """Return 1 for nucleotide transitions, else 0."""
    if len(ref) != 1 or len(alt) != 1:
        return 0
    return int(f"{ref}{alt}" in TRANSITION_PAIRS)


def map_label(clnsig: str) -> int | None:
    """Map CLNSIG to binary target; return None for ambiguous labels."""
    tokens = [t.strip() for t in re.split(r"[|,/]", clnsig) if t.strip()]

    if any(token in PATHOGENIC_LABELS for token in tokens) or clnsig in PATHOGENIC_LABELS:
        return 1
    if any(token in BENIGN_LABELS for token in tokens) or clnsig in BENIGN_LABELS:
        return 0
    return None


def load_feature_dataframe(clinvar_path: Path, sample_limit: int | None) -> pd.DataFrame:
    """Parse ClinVar VCF rows into a feature dataframe."""
    print(f"Parsing {clinvar_path}...")
    rows: list[tuple[Any, ...]] = []

    with gzip.open(clinvar_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
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

            rows.append(
                (
                    chrom,
                    int(pos),
                    ref,
                    alt,
                    _parse_first_float(info_map.get("AF_ESP")),
                    _parse_first_float(info_map.get("AF_EXAC")),
                    _parse_first_float(info_map.get("AF_TGP")),
                    _normalize_clnvc(info_map.get("CLNVC")),
                    _normalize_gene_name(info_map.get("GENEINFO")),
                    _normalize_mc(info_map.get("MC")),
                    _normalize_origin(info_map.get("ORIGIN")),
                    _is_transition(ref, alt),
                    clnsig,
                    info_map.get("CLNDN", ""),
                )
            )

            if sample_limit is not None and len(rows) >= sample_limit:
                break

    print(f"Parsed {len(rows)} variants with CLNSIG labels")
    return pd.DataFrame(
        rows,
        columns=[
            "CHROM",
            "POS",
            "REF",
            "ALT",
            "AF_ESP",
            "AF_EXAC",
            "AF_TGP",
            "CLNVC",
            "GENEINFO",
            "MC",
            "ORIGIN",
            "IS_TRANSITION",
            "CLNSIG",
            "CLNDN",
        ],
    )


def prepare_training_dataframe(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Clean labels, normalize features, and keep the model's target subset."""
    feature_df["label"] = feature_df["CLNSIG"].map(map_label)
    feature_df = feature_df.dropna(subset=["label"]).copy()
    feature_df["label"] = feature_df["label"].astype(int)

    print(f"After filtering to Pathogenic/Benign: {len(feature_df)} rows")
    print(f"Class distribution:\n{feature_df['label'].value_counts().to_string()}\n")

    # Keep SNVs only for the current model family so inference stays aligned.
    snv_mask = (feature_df["REF"].str.len() == 1) & (feature_df["ALT"].str.len() == 1)
    feature_df = feature_df[snv_mask].copy()
    print(f"After filtering to SNVs only: {len(feature_df)} rows")

    for col, fallback in (
        ("CLNVC", "UNKNOWN_CLNVC"),
        ("GENEINFO", "UNKNOWN_GENE"),
        ("MC", "UNKNOWN_MC"),
        ("ORIGIN", "UNKNOWN_ORIGIN"),
    ):
        feature_df[col] = feature_df[col].fillna(fallback).replace("", fallback)

    for col in ("AF_ESP", "AF_EXAC", "AF_TGP", "POS", "IS_TRANSITION"):
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    return feature_df


def encode_categorical_features(
    feature_df: pd.DataFrame,
    categorical_cols: tuple[str, ...],
) -> dict[str, LabelEncoder]:
    """Fit label encoders and transform categorical columns in place."""
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        feature_df[col] = encoder.fit_transform(feature_df[col])
        encoders[col] = encoder
        print(f"  {col}: {len(encoder.classes_)} classes")
    return encoders


def build_metrics(y_true, y_pred, y_proba) -> dict[str, Any]:
    """Return a compact metrics dictionary that is easy to inspect later."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Pathogenic"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_proba)), 6),
        "pathogenic_precision": round(float(report["Pathogenic"]["precision"]), 6),
        "pathogenic_recall": round(float(report["Pathogenic"]["recall"]), 6),
        "pathogenic_f1": round(float(report["Pathogenic"]["f1-score"]), 6),
        "benign_precision": round(float(report["Benign"]["precision"]), 6),
        "benign_recall": round(float(report["Benign"]["recall"]), 6),
        "benign_f1": round(float(report["Benign"]["f1-score"]), 6),
        "support_pathogenic": int(report["Pathogenic"]["support"]),
        "support_benign": int(report["Benign"]["support"]),
    }


def main() -> None:
    args = parse_args()
    clinvar_path = Path(args.clinvar_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = load_feature_dataframe(clinvar_path, args.sample_limit)
    feature_df = prepare_training_dataframe(feature_df)

    categorical_cols = ("CHROM", "REF", "ALT", "CLNVC", "GENEINFO", "MC", "ORIGIN")
    encoders = encode_categorical_features(feature_df, categorical_cols)

    feature_cols = [
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "AF_ESP",
        "AF_EXAC",
        "AF_TGP",
        "CLNVC",
        "GENEINFO",
        "MC",
        "ORIGIN",
        "IS_TRANSITION",
    ]
    X = feature_df[feature_cols].astype("float32").values
    y = feature_df["label"].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=args.random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1764705882,  # 15% of the original total
        random_state=args.random_state,
        stratify=y_train_val,
    )

    print(
        f"\nTrain: {X_train.shape[0]}  |  Validation: {X_val.shape[0]}  |  Test: {X_test.shape[0]}"
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(
        "Class balance - "
        f"Benign: {n_neg}, Pathogenic: {n_pos}, scale_pos_weight: {scale_pos_weight:.2f}"
    )

    print("\nTraining XGBClassifier...")
    model = XGBClassifier(
        n_estimators=1400,
        max_depth=8,
        learning_rate=0.03,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric=["logloss", "aucpr"],
        early_stopping_rounds=60,
        random_state=args.random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("\n--- Validation Metrics ---")
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = build_metrics(y_val, y_val_pred, y_val_proba)
    print(json.dumps(val_metrics, indent=2))

    print("\n--- Test Metrics ---")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = build_metrics(y_test, y_test_pred, y_test_proba)
    print(json.dumps(test_metrics, indent=2))

    print("\nBuilding disease_map from pathogenic variants...")
    pathogenic_df = feature_df[feature_df["label"] == 1].copy()
    for col in ("CHROM", "REF", "ALT"):
        pathogenic_df[col] = encoders[col].inverse_transform(pathogenic_df[col])

    disease_map: dict[str, str] = {}
    for _, row in pathogenic_df.iterrows():
        key = f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}"
        disease = row["CLNDN"]
        if disease and disease != "not_provided":
            disease_map[key] = disease.replace("_", " ").replace("|", ", ")

    print(f"Disease map: {len(disease_map)} pathogenic variants with known diseases")

    model_path = output_dir / "model.joblib"
    encoders_path = output_dir / "encoders.joblib"
    disease_map_path = output_dir / "disease_map.joblib"
    metrics_path = output_dir / "model_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(
        {
            "schema_version": 3,
            "feature_cols": feature_cols,
            "encoders": encoders,
        },
        encoders_path,
    )
    joblib.dump(disease_map, disease_map_path)

    metrics_payload = {
        "clinvar_path": str(clinvar_path),
        "sample_limit": args.sample_limit,
        "random_state": args.random_state,
        "feature_cols": feature_cols,
        "categorical_cols": list(categorical_cols),
        "rows_after_filtering": int(len(feature_df)),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {model_path}")
    print(f"  {encoders_path}")
    print(f"  {disease_map_path}")
    print(f"  {metrics_path}")
    print("Done. Restart the FastAPI server to use the new model.")


if __name__ == "__main__":
    main()
