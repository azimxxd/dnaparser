"""
FastAPI REST API for VCF variant risk analysis.

Accepts a VCF file upload, runs each variant through an XGBoost classifier,
and returns the top 50 most dangerous mutations sorted by risk score.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# --- PASTE YOUR FAST_VCF_PARSER FUNCTION HERE ---
#
# Expected signature:
#     def fast_vcf_parser(file_path: str, batch_size: int, include_info: bool = False)
#         -> Generator[list[dict], None, None]
#
# Each yielded batch is a list of dicts:
# {"CHROM": ..., "POS": ..., "REF": ..., "ALT": ..., optional "INFO": ...}
# ============================================================================
from vcf_parser import parse_vcf as fast_vcf_parser


# ============================================================================
# Load trained model and encoders ONCE at server startup.
# This avoids re-reading from disk on every request.
# ============================================================================
MODEL = joblib.load("model.joblib")
_encoders_blob = joblib.load("encoders.joblib")
if isinstance(_encoders_blob, dict) and "encoders" in _encoders_blob:
    # New artifact format with explicit feature ordering.
    ENCODERS: dict = _encoders_blob["encoders"]
    FEATURE_COLUMNS: list[str] = _encoders_blob.get(
        "feature_cols",
        ["CHROM", "POS", "REF", "ALT"],
    )
else:
    # Backward compatibility with old artifact format:
    # encoders.joblib is just {"CHROM": LabelEncoder, "REF": ..., "ALT": ...}.
    ENCODERS = _encoders_blob
    FEATURE_COLUMNS = ["CHROM", "POS", "REF", "ALT"]

DISEASE_MAP: dict[str, str] = joblib.load("disease_map.joblib")  # "CHROM_POS_REF_ALT" -> disease name

# Pre-compute lookup dicts for O(1) encoding. Much faster than calling
# encoder.transform() on each batch because it avoids numpy overhead.
ENCODING_MAPS: dict[str, dict[str, int]] = {}
for col_name, encoder in ENCODERS.items():
    ENCODING_MAPS[col_name] = {
        label: idx for idx, label in enumerate(encoder.classes_)
    }

# Default encoded value for categories unseen during training.
# Using -1 — XGBoost handles missing/unknown values gracefully.
UNKNOWN_CODE = -1

INFO_DERIVED_FEATURES = {"AF_ESP", "AF_EXAC", "CLNVC", "GENEINFO"}
USES_INFO_COLUMN = any(col in INFO_DERIVED_FEATURES for col in FEATURE_COLUMNS)

# Allowed upload extensions.
ALLOWED_EXTENSIONS = {".vcf", ".vcf.gz"}

# Max upload size: 2 GB.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


app = FastAPI(
    title="VCF Variant Risk Analyzer",
    description="Upload a VCF file to scan for high-risk mutations.",
)


# ============================================================================
# Validation helpers
# ============================================================================

def _validate_filename(filename: str | None) -> None:
    """Reject files that don't have a .vcf or .vcf.gz extension."""
    if not filename:
        raise _error(400, "No filename provided in the upload.")
    lower = filename.lower()
    if not (lower.endswith(".vcf") or lower.endswith(".vcf.gz")):
        raise _error(
            400,
            "Invalid file format. Please upload a .vcf file.",
        )


def _error(status_code: int, message: str, details: str | None = None) -> Exception:
    """Build a lightweight sentinel exception carrying a JSONResponse.

    We raise this from validation helpers and catch it in the endpoint
    so that every error path returns clean JSON, never an HTML traceback.
    """
    body: dict[str, Any] = {"status": "error", "message": message}
    if details:
        body["details"] = details
    return _APIError(status_code, body)


class _APIError(Exception):
    """Internal exception that carries a ready-to-send JSONResponse."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body


def _parse_info(info: str) -> dict[str, str]:
    """Parse a semicolon-delimited VCF INFO string into a key-value map."""
    if not info:
        return {}
    result: dict[str, str] = {}
    for item in info.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _parse_info_float(raw_value: str | None) -> float:
    """Parse first numeric value from INFO fields like AF_EXAC=0.01,0.02."""
    if not raw_value:
        return float(np.nan)
    first = raw_value.split(",", 1)[0].strip()
    try:
        return float(first)
    except ValueError:
        return float(np.nan)


def _normalize_gene_name(raw_value: str | None) -> str:
    """Extract a stable gene symbol from ClinVar GENEINFO."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_GENE"
    # Example: "BRCA1:672|NBR2:10230" -> "BRCA1"
    first_gene = raw_value.split("|", 1)[0]
    symbol = first_gene.split(":", 1)[0].strip()
    return symbol or "UNKNOWN_GENE"


# ============================================================================
# ML prediction
# ============================================================================

def real_ml_predict(batch: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run XGBoost inference on a batch of variants.

    Encodes categorical features using the pre-fitted LabelEncoders,
    calls model.predict_proba to get pathogenicity probability, and
    filters for dangerous mutations (risk_score > 0.80).

    Args:
        batch: List of variant dicts. For extended models the parser should
            include INFO as well.

    Returns:
        List of dangerous mutation dicts.
    """
    # Build the feature matrix as a numpy array for predict_proba.
    # Column ordering is loaded from encoders.joblib metadata.
    n = len(batch)
    X = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float32)

    for i, variant in enumerate(batch):
        info_map = _parse_info(variant.get("INFO", "")) if USES_INFO_COLUMN else {}

        for j, feature in enumerate(FEATURE_COLUMNS):
            if feature == "POS":
                try:
                    X[i, j] = float(variant["POS"])
                except (TypeError, ValueError):
                    X[i, j] = float(np.nan)
                continue

            if feature == "AF_ESP":
                X[i, j] = _parse_info_float(info_map.get("AF_ESP"))
                continue

            if feature == "AF_EXAC":
                X[i, j] = _parse_info_float(info_map.get("AF_EXAC"))
                continue

            if feature == "CLNVC":
                clnvc = info_map.get("CLNVC", "UNKNOWN_CLNVC")
                clnvc_map = ENCODING_MAPS.get("CLNVC", {})
                X[i, j] = clnvc_map.get(clnvc, UNKNOWN_CODE)
                continue

            if feature == "GENEINFO":
                gene = _normalize_gene_name(info_map.get("GENEINFO"))
                gene_map = ENCODING_MAPS.get("GENEINFO", {})
                X[i, j] = gene_map.get(gene, UNKNOWN_CODE)
                continue

            # CHROM / REF / ALT and any additional categorical features.
            if feature in ENCODING_MAPS:
                X[i, j] = ENCODING_MAPS[feature].get(variant.get(feature, ""), UNKNOWN_CODE)
                continue

            # Fallback for any future numeric passthrough features.
            try:
                X[i, j] = float(variant.get(feature, np.nan))
            except (TypeError, ValueError):
                X[i, j] = float(np.nan)

    # predict_proba returns [[p_class0, p_class1], ...].
    # Column 1 is the probability of Pathogenic (our risk_score).
    probas = MODEL.predict_proba(X)[:, 1]

    # Filter for dangerous mutations and look up associated diseases.
    dangerous: list[dict[str, Any]] = []
    for i, risk_score in enumerate(probas):
        if risk_score > 0.80:
            variant = batch[i]
            # Build the lookup key matching the format in disease_map.
            key = f"{variant['CHROM']}_{variant['POS']}_{variant['REF']}_{variant['ALT']}"
            dangerous.append(
                {
                    "chromosome": variant["CHROM"],
                    "position": variant["POS"],
                    "mutation": f"{variant['REF']} -> {variant['ALT']}",
                    "risk_score": round(float(risk_score), 6),
                    "associated_disease": DISEASE_MAP.get(key, "Novel/Unknown Pathology"),
                }
            )

    return dangerous


# ============================================================================
# Endpoint
# ============================================================================

@app.post("/analyze")
async def analyze_vcf(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a VCF file upload, scan variants, and return top risks.

    Validation order:
        1. File extension (.vcf or .vcf.gz)
        2. Non-empty body
        3. VCF format (parser checks for #CHROM header)

    Returns:
        JSON with status, total_variants_scanned, and top_risks (max 50).
    """
    tmp_path: str | None = None

    try:
        # --- 1. File extension validation ---
        _validate_filename(file.filename)

        # --- 2. Save upload to temp file ---
        suffix = ".vcf.gz" if file.filename and file.filename.lower().endswith(".gz") else ".vcf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name

        total_bytes = 0
        while chunk := await file.read(8 * 1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise _error(400, f"File exceeds maximum upload size of {MAX_UPLOAD_BYTES // (1024**3)} GB.")
            tmp.write(chunk)
        tmp.close()

        # --- 3. Empty file check ---
        if total_bytes == 0:
            raise _error(400, "Uploaded file is empty (0 bytes).")

        # --- 4. Parse and predict ---
        # The parser raises ValueError if #CHROM header is missing,
        # which we catch below and translate to a 400.
        total_scanned = 0
        all_dangerous: list[dict[str, Any]] = []

        for batch in fast_vcf_parser(tmp_path, batch_size=50_000, include_info=USES_INFO_COLUMN):
            total_scanned += len(batch)
            dangerous = real_ml_predict(batch)
            all_dangerous.extend(dangerous)

        # Sort by risk_score descending and keep top 50.
        all_dangerous.sort(key=lambda x: x["risk_score"], reverse=True)
        top_risks = all_dangerous[:50]

        return JSONResponse(
            content={
                "status": "completed",
                "total_variants_scanned": total_scanned,
                "top_risks": top_risks,
            }
        )

    except _APIError as exc:
        # Validation errors — return the pre-built JSON response.
        logger.warning("Validation error: %s", exc.body["message"])
        return JSONResponse(status_code=exc.status_code, content=exc.body)

    except ValueError as exc:
        # Parser-level format errors (missing #CHROM header, bad columns).
        logger.warning("VCF format error: %s", exc)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Invalid VCF format. The file could not be parsed.",
                "details": str(exc),
            },
        )

    except Exception as exc:
        # Catch-all for anything unexpected — log full traceback for debugging.
        logger.exception("Unexpected error during /analyze")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred during analysis.",
                "details": str(exc),
            },
        )

    finally:
        # Always clean up the temporary file, even on errors.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Run the server:
#   uvicorn main:app --host 0.0.0.0 --port 8000
#
# Test with curl:
#   curl -X POST http://localhost:8000/analyze \
#        -F "file=@sample.vcf"
#
# Or with a gzipped file:
#   curl -X POST http://localhost:8000/analyze \
#        -F "file=@clinvar.vcf.gz"
# ---------------------------------------------------------------------------
