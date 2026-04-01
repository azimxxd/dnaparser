"""
FastAPI REST API for VCF variant risk analysis.

Accepts a VCF file upload, runs each variant through an XGBoost classifier,
and returns the top 50 most dangerous mutations sorted by risk score.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
if os.path.exists("model_metrics.json"):
    with open("model_metrics.json", encoding="utf-8") as fh:
        MODEL_METRICS: dict[str, Any] | None = json.load(fh)
else:
    MODEL_METRICS = None

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

INFO_DERIVED_FEATURES = {"AF_ESP", "AF_EXAC", "AF_TGP", "CLNVC", "GENEINFO", "MC", "ORIGIN"}
USES_INFO_COLUMN = any(col in INFO_DERIVED_FEATURES for col in FEATURE_COLUMNS)
DANGEROUS_THRESHOLD = 0.80
TOP_RISKS_LIMIT = 50
TOP_CANDIDATES_LIMIT = 10
TRANSITION_PAIRS = {"AG", "GA", "CT", "TC"}
ORIGIN_LABELS = {
    "0": "unknown",
    "1": "germline",
    "2": "somatic",
    "4": "inherited",
    "8": "paternal",
    "16": "maternal",
    "32": "de_novo",
    "64": "biparental",
    "128": "uniparental",
    "256": "not_tested",
    "512": "tested_inconclusive",
    "1073741824": "other",
}
SPECIALTY_KEYWORD_RULES = (
    (
        "ophthalmology_inherited_retinal",
        "Ophthalmology / inherited retinal disease genetics",
        ("retina", "retinal", "macular", "stargardt", "cone dystrophy", "rod dystrophy", "vision"),
    ),
    (
        "oncology_cancer_genetics",
        "Oncology / cancer genetics",
        ("cancer", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma", "sarcoma", "melanoma"),
    ),
    (
        "neurology_neurogenetics",
        "Neurology / neurogenetics",
        ("neurolog", "seizure", "epilep", "ataxia", "neurodevelopment", "encephal", "cognitive", "developmental"),
    ),
    (
        "cardiology_cardiovascular_genetics",
        "Cardiology / cardiovascular genetics",
        ("cardio", "arrhythm", "heart", "qt", "myocard", "cardiomyopathy"),
    ),
    (
        "clinical_immunology",
        "Clinical immunology / genetics",
        ("immune", "immunodeficiency", "mycobacterial"),
    ),
    (
        "metabolic_genetics",
        "Metabolic genetics",
        ("metabolic", "peroxisome", "mitochond"),
    ),
    (
        "hearing_otology_genetics",
        "Audiology / otolaryngology genetics",
        ("hearing", "deaf", "auditory"),
    ),
)

# Allowed upload extensions.
ALLOWED_EXTENSIONS = {".vcf", ".vcf.gz"}

# Max upload size: 2 GB.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
REPORTS_DIR = os.getenv("REPORTS_DIR", "saved_reports")
REPORT_SCHEMA_VERSION = 1
FRONTEND_DIR = Path("frontend")
FRONTEND_ASSETS_DIR = FRONTEND_DIR / "assets"
DEMO_CASES_DIR = Path("demo_cases")


app = FastAPI(
    title="VCF Variant Risk Analyzer",
    description="Upload a VCF file to scan for high-risk mutations.",
)
app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")


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


def _normalize_clnvc(raw_value: str | None) -> str:
    """Normalize variant class into a stable category."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_CLNVC"
    return raw_value.strip() or "UNKNOWN_CLNVC"


def _normalize_mc(raw_value: str | None) -> str:
    """Extract the first molecular consequence label from MC."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_MC"
    first = raw_value.split(",", 1)[0]
    if "|" in first:
        consequence = first.split("|", 1)[1].strip()
        return consequence or "UNKNOWN_MC"
    return first.strip() or "UNKNOWN_MC"


def _normalize_origin(raw_value: str | None) -> str:
    """Keep the first ORIGIN code so it remains a stable category."""
    if not raw_value or raw_value == ".":
        return "UNKNOWN_ORIGIN"
    return raw_value.split(",", 1)[0].strip() or "UNKNOWN_ORIGIN"


def _is_transition(ref: str, alt: str) -> float:
    """Return 1.0 for A<->G / C<->T substitutions, else 0.0."""
    if len(ref) != 1 or len(alt) != 1:
        return 0.0
    return float(f"{ref}{alt}" in TRANSITION_PAIRS)


def _describe_origin(origin_code: str | None) -> str | None:
    """Turn compact ORIGIN codes into human-readable labels when possible."""
    if not origin_code or origin_code == "UNKNOWN_ORIGIN":
        return None
    label = ORIGIN_LABELS.get(origin_code)
    if label:
        return f"{origin_code} ({label})"
    return origin_code


def _variant_key(variant: dict[str, str]) -> str:
    """Build the lookup key used in disease_map.joblib."""
    return f"{variant['CHROM']}_{variant['POS']}_{variant['REF']}_{variant['ALT']}"


def _alert_level(risk_score: float) -> str:
    """Bucket raw score into a simple label for API consumers."""
    if risk_score >= 0.95:
        return "very_high"
    if risk_score > DANGEROUS_THRESHOLD:
        return "high"
    if risk_score >= 0.50:
        return "medium"
    return "low"


def _supported_variant_scope() -> dict[str, Any]:
    """Describe what the current model is intended to support."""
    return {
        "variant_types": ["SNV"],
        "accepts_file_extensions": sorted(ALLOWED_EXTENSIONS),
        "uses_info_fields": sorted(INFO_DERIVED_FEATURES.intersection(FEATURE_COLUMNS)),
        "notes": [
            "The current model was trained only on single-nucleotide variants (SNVs).",
            "The API can parse broader VCF content, but the score is designed for SNVs.",
            "Risk scores are for prioritization, not diagnosis.",
        ],
    }


def _build_model_info() -> dict[str, Any]:
    """Return a compact description of the loaded model and its training stats."""
    encoder_schema_version = None
    if isinstance(_encoders_blob, dict):
        encoder_schema_version = _encoders_blob.get("schema_version")

    model_info: dict[str, Any] = {
        "status": "ready",
        "model_type": type(MODEL).__name__,
        "encoder_schema_version": encoder_schema_version,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_feature_columns": sorted(ENCODERS.keys()),
        "dangerous_threshold": DANGEROUS_THRESHOLD,
        "top_risks_limit": TOP_RISKS_LIMIT,
        "top_candidates_limit": TOP_CANDIDATES_LIMIT,
        "disease_map_entries": len(DISEASE_MAP),
        "supported_scope": _supported_variant_scope(),
    }
    if MODEL_METRICS is not None:
        model_info["training_summary"] = MODEL_METRICS
    return model_info


def _readiness_checks() -> dict[str, bool]:
    """Return lightweight readiness checks for the running service."""
    return {
        "model_loaded": MODEL is not None,
        "encoders_loaded": bool(ENCODERS),
        "feature_columns_loaded": bool(FEATURE_COLUMNS),
        "disease_map_loaded": bool(DISEASE_MAP),
        "metrics_loaded": MODEL_METRICS is not None,
    }


def _most_common_labels(counter: Counter[str], limit: int = 3) -> list[dict[str, Any]]:
    """Convert a Counter into a compact JSON-friendly top-N list."""
    return [
        {"label": label, "count": count}
        for label, count in counter.most_common(limit)
        if label
    ]


def _build_analysis_summary(
    *,
    total_scanned: int,
    dangerous_count: int,
    max_raw_score: float,
    alert_level_counts: Counter[str],
    dangerous_gene_counts: Counter[str],
    dangerous_mc_counts: Counter[str],
    candidate_gene_counts: Counter[str],
    candidate_mc_counts: Counter[str],
    top_risks: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a file-level summary that is easy to read without a UI."""
    if dangerous_count > 0:
        overall_alert_level = "very_high" if max_raw_score >= 0.95 else "high"
        top_genes = _most_common_labels(dangerous_gene_counts)
        top_molecular_consequences = _most_common_labels(dangerous_mc_counts)
        if dangerous_count == 1:
            short_text = "1 variant crossed the alert threshold in this file."
        else:
            short_text = f"{dangerous_count} variants crossed the alert threshold in this file."
        if top_genes:
            short_text += f" Top gene signal: {top_genes[0]['label']}."
    elif max_raw_score >= 0.50:
        overall_alert_level = "medium"
        top_genes = _most_common_labels(candidate_gene_counts)
        top_molecular_consequences = _most_common_labels(candidate_mc_counts)
        short_text = (
            "No variant crossed the alert threshold, but at least one candidate "
            "reached a medium model score."
        )
        if top_candidates:
            short_text += (
                f" Best candidate score: {top_candidates[0]['risk_score']:.6f}."
            )
    else:
        overall_alert_level = "low"
        top_genes = _most_common_labels(candidate_gene_counts)
        top_molecular_consequences = _most_common_labels(candidate_mc_counts)
        short_text = (
            "No variant crossed the alert threshold and the strongest candidate "
            "stayed in the low-score range."
        )

    return {
        "overall_alert_level": overall_alert_level,
        "short_text": short_text,
        "total_variants_scanned": total_scanned,
        "dangerous_variants_found": dangerous_count,
        "dangerous_variant_rate": round(dangerous_count / total_scanned, 6) if total_scanned else 0.0,
        "max_raw_score": round(max_raw_score, 6),
        "alert_level_counts": {
            "very_high": int(alert_level_counts.get("very_high", 0)),
            "high": int(alert_level_counts.get("high", 0)),
            "medium": int(alert_level_counts.get("medium", 0)),
            "low": int(alert_level_counts.get("low", 0)),
        },
        "top_genes": top_genes,
        "top_molecular_consequences": top_molecular_consequences,
        "reference_hits_in_top_risks": sum(
            1 for item in top_risks if item["associated_disease"] != "Novel/Unknown Pathology"
        ),
        "reference_hits_in_top_candidates": sum(
            1 for item in top_candidates if item["associated_disease"] != "Novel/Unknown Pathology"
        ),
    }


def _infer_specialist_context(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infer possible specialist areas from disease-name keywords.

    This is intentionally heuristic: it suggests review contexts, not diagnoses.
    """
    matches: Counter[str] = Counter()

    for item in results:
        disease = str(item.get("associated_disease", "")).lower()
        if not disease or disease == "novel/unknown pathology":
            continue
        for rule_id, _, keywords in SPECIALTY_KEYWORD_RULES:
            if any(keyword in disease for keyword in keywords):
                matches[rule_id] += 1

    specialist_index = {rule_id: label for rule_id, label, _ in SPECIALTY_KEYWORD_RULES}
    return [
        {
            "id": rule_id,
            "label": specialist_index[rule_id],
            "matched_results": count,
        }
        for rule_id, count in matches.most_common(3)
    ]


def _build_follow_up_guidance(
    analysis_summary: dict[str, Any],
    top_risks: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return safe next-step guidance based on the current analysis output."""
    overall_alert_level = analysis_summary["overall_alert_level"]
    top_genes = analysis_summary["top_genes"]
    top_consequences = analysis_summary["top_molecular_consequences"]
    reference_hits = analysis_summary["reference_hits_in_top_risks"]

    steps: list[str] = []
    if overall_alert_level in {"very_high", "high"}:
        steps.extend(
            [
                "Review the flagged variants with a clinical geneticist or molecular diagnostics specialist.",
                "Consider orthogonal confirmation of the top variants before any clinical decision.",
                "Correlate the flagged variants with phenotype, symptoms, family history, and the original laboratory context.",
            ]
        )
        if reference_hits:
            steps.append("Prioritize variants that already have a disease match in the current reference map.")
        else:
            steps.append("Treat the signal as suspicious, but keep in mind that the top variants do not yet have a direct reference-map match.")
    elif overall_alert_level == "medium":
        steps.extend(
            [
                "Review the strongest candidate variants manually before discarding the file as low risk.",
                "Check whether the top candidate genes fit the clinical picture and sequencing context.",
                "Consider follow-up interpretation if phenotype or family history strongly points to a genetic condition.",
            ]
        )
    else:
        steps.extend(
            [
                "No high-alert variants were found in this file.",
                "If clinical suspicion remains high, review the raw VCF and the model's top candidates manually.",
                "Remember that this pipeline focuses on SNVs and does not rule out disease outside that scope.",
            ]
        )

    if top_genes:
        steps.append(f"Start the manual review around the strongest repeated gene signal: {top_genes[0]['label']}.")
    if top_consequences:
        steps.append(
            "Pay extra attention to the main consequence class in the strongest findings: "
            f"{top_consequences[0]['label']}."
        )

    specialist_context = _infer_specialist_context(top_risks or top_candidates)
    cautions = [
        "This output prioritizes variants; it does not establish a diagnosis or treatment plan.",
        "Variant interpretation should be combined with phenotype, family history, and laboratory review.",
        "The current model is designed for SNVs, so broader variant classes need separate interpretation.",
    ]
    if specialist_context:
        cautions.append(
            "Suggested specialist areas are heuristic and inferred from disease-name keywords, not from a dedicated clinical recommendation engine."
        )

    return {
        "urgency": overall_alert_level,
        "recommended_next_steps": steps,
        "possible_specialist_context": specialist_context,
        "clinical_caution": cautions,
    }


def _model_signal_label(risk_score: float) -> str:
    """Turn the raw score into a simple strength bucket."""
    if risk_score >= 0.95:
        return "very_strong"
    if risk_score > DANGEROUS_THRESHOLD:
        return "strong"
    if risk_score >= 0.50:
        return "moderate"
    if risk_score >= 0.05:
        return "weak"
    return "minimal"


def _population_signal_label(
    af_esp: float | None,
    af_exac: float | None,
    af_tgp: float | None,
) -> str:
    """Summarize the population-frequency evidence in a compact label."""
    known_freqs = [value for value in (af_esp, af_exac, af_tgp) if value is not None]
    if not known_freqs:
        return "missing"
    min_freq = min(known_freqs)
    max_freq = max(known_freqs)
    if min_freq <= 0.001:
        return "very_low_frequency"
    if max_freq >= 0.05:
        return "high_frequency"
    return "present_but_not_extreme"


def _consequence_signal_label(mc: str | None) -> str:
    """Group molecular consequences into broad evidence buckets."""
    if not mc:
        return "unknown"

    lowered = mc.lower()
    high_impact_keywords = (
        "splice_acceptor",
        "splice_donor",
        "frameshift",
        "stop_gained",
        "stop_lost",
        "start_lost",
    )
    moderate_impact_keywords = (
        "missense",
        "protein_altering",
        "inframe",
    )
    low_impact_keywords = (
        "synonymous",
        "intergenic",
        "upstream",
        "downstream",
        "non_coding_transcript",
        "intron",
    )

    if any(keyword in lowered for keyword in high_impact_keywords):
        return "high_impact"
    if any(keyword in lowered for keyword in moderate_impact_keywords):
        return "moderate_impact"
    if any(keyword in lowered for keyword in low_impact_keywords):
        return "low_impact"
    return "other"


def _build_evidence_profile(
    *,
    risk_score: float,
    disease: str,
    gene: str | None,
    molecular_consequence: str | None,
    af_esp: float | None,
    af_exac: float | None,
    af_tgp: float | None,
) -> tuple[dict[str, Any], str, str, bool]:
    """Summarize why the result looks trustworthy or contradictory."""
    reference_match = disease != "Novel/Unknown Pathology"
    model_signal = _model_signal_label(risk_score)
    population_signal = _population_signal_label(af_esp, af_exac, af_tgp)
    consequence_signal = _consequence_signal_label(molecular_consequence)
    reference_model_conflict = reference_match and risk_score < 0.50
    novel_high_signal_candidate = (not reference_match) and risk_score > DANGEROUS_THRESHOLD

    support_score = 0
    if model_signal == "very_strong":
        support_score += 3
    elif model_signal == "strong":
        support_score += 2
    elif model_signal == "moderate":
        support_score += 1

    if reference_match:
        support_score += 1
    if consequence_signal == "high_impact":
        support_score += 1
    elif consequence_signal == "low_impact":
        support_score -= 1

    if population_signal == "very_low_frequency":
        support_score += 1
    elif population_signal == "high_frequency":
        support_score -= 1

    if reference_model_conflict:
        support_score -= 2

    if support_score >= 4:
        confidence_level = "very_high"
    elif support_score >= 3:
        confidence_level = "high"
    elif support_score >= 2:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    if reference_model_conflict:
        confidence_reason = (
            "The exact variant exists in the reference map, but the current model signal stays low."
        )
    elif novel_high_signal_candidate:
        confidence_reason = (
            "The model signal is strong even without an exact reference-map match, so this looks like a novel high-priority candidate."
        )
    elif confidence_level in {"very_high", "high"}:
        confidence_reason = (
            "The model score is supported by the current evidence profile and does not show an obvious contradiction."
        )
    elif confidence_level == "medium":
        confidence_reason = (
            "Some evidence supports the finding, but the overall picture is mixed."
        )
    else:
        confidence_reason = (
            "The evidence profile is weak or internally mixed, so the finding should be treated cautiously."
        )

    evidence_profile = {
        "model_signal": model_signal,
        "reference_match": reference_match,
        "reference_model_conflict": reference_model_conflict,
        "novel_high_signal_candidate": novel_high_signal_candidate,
        "population_signal": population_signal,
        "consequence_signal": consequence_signal,
        "gene_present": bool(gene),
    }
    return evidence_profile, confidence_level, confidence_reason, reference_model_conflict


def _collect_unique_diseases(results: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Keep the first unique non-empty disease labels from ranked results."""
    seen: list[str] = []
    for item in results:
        disease = str(item.get("associated_disease", "")).strip()
        if not disease or disease == "Novel/Unknown Pathology":
            continue
        if disease not in seen:
            seen.append(disease)
        if len(seen) >= limit:
            break
    return seen


def _build_clinical_report(
    source_filename: str,
    analysis_report: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured mini-report for downstream apps or demos."""
    analysis_summary = analysis_report["analysis_summary"]
    follow_up_guidance = analysis_report["follow_up_guidance"]
    top_risks = analysis_report["top_risks"]
    top_candidates = analysis_report["top_candidates"]
    highlighted_results = top_risks[:5] if top_risks else top_candidates[:5]

    overall_alert_level = str(analysis_summary["overall_alert_level"])
    dangerous_count = int(analysis_summary["dangerous_variants_found"])
    total_scanned = int(analysis_summary["total_variants_scanned"])
    top_genes = analysis_summary["top_genes"]
    top_consequences = analysis_summary["top_molecular_consequences"]
    unique_diseases = _collect_unique_diseases(highlighted_results)

    if overall_alert_level in {"very_high", "high"}:
        headline = f"High-priority variant review is recommended for {source_filename}."
        what_this_means = (
            "The file contains variants that strongly resemble known pathogenic patterns "
            "from the model's training data."
        )
    elif overall_alert_level == "medium":
        headline = f"Manual variant review is recommended for {source_filename}."
        what_this_means = (
            "No variant crossed the alert threshold, but the strongest candidates still "
            "deserve a closer look."
        )
    else:
        headline = f"No high-alert variant signal was detected in {source_filename}."
        what_this_means = (
            "The strongest variants in this file stayed in the low-score range under the "
            "current SNV-focused model."
        )

    if dangerous_count == 1:
        threshold_sentence = "1 variant crossed the current alert threshold."
    elif dangerous_count > 1:
        threshold_sentence = f"{dangerous_count} variants crossed the current alert threshold."
    else:
        threshold_sentence = "No variant crossed the current alert threshold."

    plain_language_summary = [
        f"The file was scanned for {total_scanned} variants.",
        threshold_sentence,
        analysis_summary["short_text"],
        what_this_means,
    ]

    if top_genes:
        plain_language_summary.append(
            f"The strongest repeated gene signal in this file is {top_genes[0]['label']}."
        )
    if top_consequences:
        plain_language_summary.append(
            "The main consequence pattern among the strongest findings is "
            f"{top_consequences[0]['label']}."
        )

    main_findings = [
        {
            "rank": rank,
            "gene": item.get("gene"),
            "variant": f"{item['chromosome']}:{item['position']} {item['mutation']}",
            "risk_score": item["risk_score"],
            "alert_level": item["alert_level"],
            "confidence_level": item["confidence_level"],
            "reference_model_conflict": item["reference_model_conflict"],
            "associated_disease": item["associated_disease"],
            "molecular_consequence": item.get("molecular_consequence"),
            "evidence_profile": item["evidence_profile"],
            "why_it_matters": item["explanation"],
        }
        for rank, item in enumerate(highlighted_results, start=1)
    ]

    watchlist = []
    if not top_risks:
        watchlist = [
            {
                "rank": rank,
                "gene": item.get("gene"),
                "variant": f"{item['chromosome']}:{item['position']} {item['mutation']}",
                "risk_score": item["risk_score"],
                "alert_level": item["alert_level"],
                "confidence_level": item["confidence_level"],
                "reference_model_conflict": item["reference_model_conflict"],
                "evidence_profile": item["evidence_profile"],
                "reason_to_watch": item["explanation"],
            }
            for rank, item in enumerate(top_candidates[:3], start=1)
        ]

    limitations = list(
        dict.fromkeys(
            follow_up_guidance["clinical_caution"]
            + _supported_variant_scope()["notes"]
        )
    )

    return {
        "report_type": "variant_triage_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_filename,
        "headline": headline,
        "overview": {
            "overall_alert_level": overall_alert_level,
            "urgency": follow_up_guidance["urgency"],
            "total_variants_scanned": total_scanned,
            "dangerous_variants_found": dangerous_count,
            "dangerous_variant_rate": analysis_summary["dangerous_variant_rate"],
            "max_raw_score": analysis_summary["max_raw_score"],
            "threshold_used": analysis_report["threshold_used"],
        },
        "plain_language_summary": plain_language_summary,
        "interpretation": {
            "what_stands_out": analysis_summary["short_text"],
            "what_this_means": what_this_means,
            "focus_genes": top_genes,
            "focus_molecular_consequences": top_consequences,
            "known_disease_context": unique_diseases,
            "reference_hits": {
                "top_risks": analysis_summary["reference_hits_in_top_risks"],
                "top_candidates": analysis_summary["reference_hits_in_top_candidates"],
            },
        },
        "main_findings": main_findings,
        "candidate_watchlist": watchlist,
        "recommended_follow_up": {
            "urgency": follow_up_guidance["urgency"],
            "recommended_next_steps": follow_up_guidance["recommended_next_steps"],
            "possible_specialist_context": follow_up_guidance["possible_specialist_context"],
        },
        "limitations": limitations,
    }


def _clean_optional_float(value: float) -> float | None:
    """Convert NaN-like values to None so JSON output stays clean."""
    if np.isnan(value):
        return None
    return round(float(value), 6)


def _build_explanation(
    variant: dict[str, str],
    risk_score: float,
    disease: str,
) -> tuple[
    str,
    list[str],
    str | None,
    str | None,
    str | None,
    str | None,
    dict[str, float | None],
]:
    """Create a short human-readable explanation for a scored variant."""
    info_map = _parse_info(variant.get("INFO", "")) if USES_INFO_COLUMN else {}
    af_esp_raw = _parse_info_float(info_map.get("AF_ESP"))
    af_exac_raw = _parse_info_float(info_map.get("AF_EXAC"))
    af_tgp_raw = _parse_info_float(info_map.get("AF_TGP"))
    af_esp = _clean_optional_float(af_esp_raw)
    af_exac = _clean_optional_float(af_exac_raw)
    af_tgp = _clean_optional_float(af_tgp_raw)

    clnvc = _normalize_clnvc(info_map.get("CLNVC")) if USES_INFO_COLUMN else None
    if clnvc == "UNKNOWN_CLNVC":
        clnvc = None

    gene = _normalize_gene_name(info_map.get("GENEINFO")) if USES_INFO_COLUMN else None
    if gene == "UNKNOWN_GENE":
        gene = None

    mc = _normalize_mc(info_map.get("MC")) if USES_INFO_COLUMN else None
    if mc == "UNKNOWN_MC":
        mc = None

    origin_code = _normalize_origin(info_map.get("ORIGIN")) if USES_INFO_COLUMN else None
    origin = _describe_origin(origin_code)

    signals: list[str] = []
    if risk_score > DANGEROUS_THRESHOLD:
        signals.append("Score is above the current alert threshold.")
    else:
        signals.append("Score stays below the current alert threshold.")

    if disease != "Novel/Unknown Pathology":
        signals.append("Exact variant match was found in the current disease reference map.")
    else:
        signals.append("No exact disease match was found in the current disease reference map.")

    known_freqs = [value for value in (af_esp, af_exac, af_tgp) if value is not None]
    if known_freqs:
        min_freq = min(known_freqs)
        max_freq = max(known_freqs)
        if min_freq <= 0.001:
            signals.append("Population frequency looks very low in the provided AF fields.")
        elif max_freq >= 0.05:
            signals.append("Population frequency looks high in the provided AF fields, which usually lowers risk.")
        else:
            signals.append("Population frequency is present but not extremely low.")
    elif USES_INFO_COLUMN:
        signals.append("Population frequency fields are missing, so the score leans more on location and categorical signals.")

    if clnvc == "single_nucleotide_variant":
        signals.append("Variant class is single_nucleotide_variant, which matches the model's main training setup.")
    elif clnvc:
        signals.append(f"Variant class from INFO is {clnvc}.")

    if gene:
        signals.append(f"Gene signal from INFO is {gene}.")

    if mc:
        signals.append(f"Molecular consequence from INFO is {mc}.")

    if origin:
        signals.append(f"Origin signal from INFO is {origin}.")

    if _is_transition(variant["REF"], variant["ALT"]) == 1.0:
        signals.append("Base change is a transition substitution.")
    else:
        signals.append("Base change is a transversion substitution.")

    if risk_score >= 0.95:
        summary = "Very strong model signal for this variant."
    elif risk_score > DANGEROUS_THRESHOLD:
        summary = "Strong model signal; this variant is flagged as dangerous."
    elif risk_score >= 0.50:
        summary = "Medium model signal; this variant is notable but below the alert threshold."
    else:
        summary = "Low model signal; this variant is below the alert threshold."

    if disease != "Novel/Unknown Pathology":
        summary += " The exact variant also exists in the current disease map."

    return summary, signals, gene, clnvc, mc, origin, {
        "AF_ESP": af_esp,
        "AF_EXAC": af_exac,
        "AF_TGP": af_tgp,
    }


def _build_result_item(variant: dict[str, str], risk_score: float) -> dict[str, Any]:
    """Convert a raw variant plus score into the API/CLI response format."""
    disease = DISEASE_MAP.get(_variant_key(variant), "Novel/Unknown Pathology")
    summary, signals, gene, clnvc, mc, origin, population_frequencies = _build_explanation(
        variant,
        risk_score,
        disease,
    )
    raw_score = float(risk_score)
    score = round(raw_score, 6)
    evidence_profile, confidence_level, confidence_reason, reference_model_conflict = _build_evidence_profile(
        risk_score=raw_score,
        disease=disease,
        gene=gene,
        molecular_consequence=mc,
        af_esp=population_frequencies["AF_ESP"],
        af_exac=population_frequencies["AF_EXAC"],
        af_tgp=population_frequencies["AF_TGP"],
    )
    return {
        "chromosome": variant["CHROM"],
        "position": variant["POS"],
        "mutation": f"{variant['REF']} -> {variant['ALT']}",
        "risk_score": score,
        "passes_threshold": raw_score > DANGEROUS_THRESHOLD,
        "alert_level": _alert_level(raw_score),
        "associated_disease": disease,
        "gene": gene,
        "variant_class": clnvc,
        "molecular_consequence": mc,
        "origin": origin,
        "is_transition": bool(_is_transition(variant["REF"], variant["ALT"])),
        "population_frequencies": population_frequencies,
        "explanation": summary,
        "key_signals": signals,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "reference_model_conflict": reference_model_conflict,
        "evidence_profile": evidence_profile,
    }


def _push_top_variant(
    heap: list[tuple[float, int, dict[str, str]]],
    order_counter: Any,
    score: float,
    variant: dict[str, str],
    limit: int,
) -> None:
    """Keep only the highest-scoring variants in a bounded heap."""
    entry = (float(score), next(order_counter), dict(variant))
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    if entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _heap_to_results(heap: list[tuple[float, int, dict[str, str]]]) -> list[dict[str, Any]]:
    """Convert a bounded heap back into sorted API-ready result rows."""
    return [
        _build_result_item(variant, score)
        for score, _, variant in sorted(heap, key=lambda item: item[0], reverse=True)
    ]


# ============================================================================
# ML prediction
# ============================================================================

def _build_feature_matrix(batch: list[dict[str, str]]) -> np.ndarray:
    """Encode a batch of raw variant dicts into the model's feature matrix."""
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

            if feature == "AF_TGP":
                X[i, j] = _parse_info_float(info_map.get("AF_TGP"))
                continue

            if feature == "CLNVC":
                clnvc = _normalize_clnvc(info_map.get("CLNVC"))
                clnvc_map = ENCODING_MAPS.get("CLNVC", {})
                X[i, j] = clnvc_map.get(clnvc, UNKNOWN_CODE)
                continue

            if feature == "GENEINFO":
                gene = _normalize_gene_name(info_map.get("GENEINFO"))
                gene_map = ENCODING_MAPS.get("GENEINFO", {})
                X[i, j] = gene_map.get(gene, UNKNOWN_CODE)
                continue

            if feature == "MC":
                mc = _normalize_mc(info_map.get("MC"))
                mc_map = ENCODING_MAPS.get("MC", {})
                X[i, j] = mc_map.get(mc, UNKNOWN_CODE)
                continue

            if feature == "ORIGIN":
                origin = _normalize_origin(info_map.get("ORIGIN"))
                origin_map = ENCODING_MAPS.get("ORIGIN", {})
                X[i, j] = origin_map.get(origin, UNKNOWN_CODE)
                continue

            if feature == "IS_TRANSITION":
                X[i, j] = _is_transition(variant["REF"], variant["ALT"])
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

    return X


def score_batch(batch: list[dict[str, str]]) -> np.ndarray:
    """Return raw pathogenicity probabilities for each variant in a batch."""
    if not batch:
        return np.array([], dtype=np.float32)
    X = _build_feature_matrix(batch)
    # predict_proba returns [[p_class0, p_class1], ...].
    # Column 1 is the probability of Pathogenic (our risk_score).
    return MODEL.predict_proba(X)[:, 1]


def real_ml_predict(batch: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run XGBoost inference on a batch of variants.

    Encodes categorical features using the pre-fitted LabelEncoders,
    calls model.predict_proba to get pathogenicity probability, and
    filters for dangerous mutations (risk_score > DANGEROUS_THRESHOLD).

    Args:
        batch: List of variant dicts. For extended models the parser should
            include INFO as well.

    Returns:
        List of dangerous mutation dicts.
    """
    probas = score_batch(batch)

    # Filter for dangerous mutations and look up associated diseases.
    dangerous: list[dict[str, Any]] = []
    for i, risk_score in enumerate(probas):
        if risk_score > DANGEROUS_THRESHOLD:
            dangerous.append(_build_result_item(batch[i], float(risk_score)))

    return dangerous


def analyze_vcf_path(
    path: str | os.PathLike[str],
    batch_size: int = 50_000,
    top_risks_limit: int = TOP_RISKS_LIMIT,
    top_candidates_limit: int = TOP_CANDIDATES_LIMIT,
) -> dict[str, Any]:
    """Analyze a VCF file path and return a compact summary for API/CLI use."""
    total_scanned = 0
    dangerous_count = 0
    max_raw_score = 0.0

    dangerous_heap: list[tuple[float, int, dict[str, str]]] = []
    candidate_heap: list[tuple[float, int, dict[str, str]]] = []
    dangerous_counter = count()
    candidate_counter = count()
    alert_level_counts: Counter[str] = Counter()
    dangerous_gene_counts: Counter[str] = Counter()
    dangerous_mc_counts: Counter[str] = Counter()

    for batch in fast_vcf_parser(path, batch_size=batch_size, include_info=USES_INFO_COLUMN):
        total_scanned += len(batch)
        scores = score_batch(batch)

        for variant, risk_score_raw in zip(batch, scores):
            risk_score = float(risk_score_raw)
            if risk_score > max_raw_score:
                max_raw_score = risk_score

            alert_level_counts[_alert_level(risk_score)] += 1

            _push_top_variant(
                candidate_heap,
                candidate_counter,
                risk_score,
                variant,
                top_candidates_limit,
            )

            if risk_score > DANGEROUS_THRESHOLD:
                dangerous_count += 1
                info_map = _parse_info(variant.get("INFO", "")) if USES_INFO_COLUMN else {}
                gene = _normalize_gene_name(info_map.get("GENEINFO")) if USES_INFO_COLUMN else "UNKNOWN_GENE"
                mc = _normalize_mc(info_map.get("MC")) if USES_INFO_COLUMN else "UNKNOWN_MC"
                if gene != "UNKNOWN_GENE":
                    dangerous_gene_counts[gene] += 1
                if mc != "UNKNOWN_MC":
                    dangerous_mc_counts[mc] += 1
                _push_top_variant(
                    dangerous_heap,
                    dangerous_counter,
                    risk_score,
                    variant,
                    top_risks_limit,
                )

    top_risks = _heap_to_results(dangerous_heap)
    top_candidates = _heap_to_results(candidate_heap)
    candidate_gene_counts = Counter(
        item["gene"] for item in top_candidates if item.get("gene")
    )
    candidate_mc_counts = Counter(
        item["molecular_consequence"]
        for item in top_candidates
        if item.get("molecular_consequence")
    )

    analysis_summary = _build_analysis_summary(
        total_scanned=total_scanned,
        dangerous_count=dangerous_count,
        max_raw_score=max_raw_score,
        alert_level_counts=alert_level_counts,
        dangerous_gene_counts=dangerous_gene_counts,
        dangerous_mc_counts=dangerous_mc_counts,
        candidate_gene_counts=candidate_gene_counts,
        candidate_mc_counts=candidate_mc_counts,
        top_risks=top_risks,
        top_candidates=top_candidates,
    )

    return {
        "total_variants_scanned": total_scanned,
        "dangerous_variants_found": dangerous_count,
        "threshold_used": DANGEROUS_THRESHOLD,
        "max_raw_score": round(max_raw_score, 6),
        "top_risks": top_risks,
        "top_candidates": top_candidates,
        "analysis_summary": analysis_summary,
        "follow_up_guidance": _build_follow_up_guidance(
            analysis_summary,
            top_risks,
            top_candidates,
        ),
    }


def _ensure_reports_dir() -> None:
    """Create the report storage directory if it does not exist yet."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _build_report_id() -> str:
    """Generate a compact stable id for saved report payloads."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"report_{timestamp}_{uuid4().hex[:10]}"


def _report_file_path(report_id: str) -> str:
    """Resolve a saved-report path from a validated report id."""
    return os.path.join(REPORTS_DIR, f"{report_id}.json")


def _validate_report_id(report_id: str) -> None:
    """Reject malformed report ids before touching the filesystem."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not report_id or any(ch not in allowed for ch in report_id):
        raise _error(400, "Invalid report_id format.")


def _save_report_payload(
    *,
    source_filename: str,
    analysis_report: dict[str, Any],
    clinical_report: dict[str, Any],
) -> dict[str, Any]:
    """Persist a report payload to disk and return storage metadata."""
    _ensure_reports_dir()
    report_id = _build_report_id()
    saved_at_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "saved_at_utc": saved_at_utc,
        "source_file": source_filename,
        "analysis_summary": analysis_report["analysis_summary"],
        "follow_up_guidance": analysis_report["follow_up_guidance"],
        "top_risks": analysis_report["top_risks"],
        "top_candidates": analysis_report["top_candidates"],
        "clinical_report": clinical_report,
    }
    file_path = _report_file_path(report_id)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return {
        "report_id": report_id,
        "saved_at_utc": saved_at_utc,
        "report_endpoint": f"/reports/{report_id}",
        "storage_backend": "filesystem_json",
    }


def _load_saved_report(report_id: str) -> dict[str, Any]:
    """Load a previously saved report payload from disk."""
    _validate_report_id(report_id)
    file_path = _report_file_path(report_id)
    if not os.path.exists(file_path):
        raise _error(404, "Report not found.", details=f"report_id={report_id}")
    with open(file_path, encoding="utf-8") as fh:
        return json.load(fh)


def _demo_case_catalog() -> list[dict[str, str]]:
    """Return lightweight metadata for built-in demo cases."""
    descriptions = {
        "high_signal": "Сильный тревожный кейс с явным pathogenic-like сигналом.",
        "low_signal": "Спокойный кейс с низким score и без high-risk вариантов.",
        "mixed_signal": "Смешанный кейс, где модель должна отделить сильное от слабого.",
    }
    preferred_order = {
        "high_signal": 0,
        "mixed_signal": 1,
        "low_signal": 2,
    }

    items: list[dict[str, str]] = []
    for path in DEMO_CASES_DIR.glob("*.vcf"):
        case_id = path.stem
        label = case_id.replace("_", " ").title()
        items.append(
            {
                "id": case_id,
                "filename": path.name,
                "label": label,
                "description": descriptions.get(case_id, "Встроенный demo-файл для быстрого прогона."),
            }
        )

    return sorted(items, key=lambda item: (preferred_order.get(item["id"], 99), item["label"]))


def _resolve_demo_case_path(case_id: str) -> Path:
    """Map a demo case id to an existing VCF file."""
    _validate_report_id(case_id)
    for path in DEMO_CASES_DIR.glob("*.vcf"):
        if path.stem == case_id:
            return path
    raise _error(404, "Demo case not found.", details=f"case_id={case_id}")


def _run_demo_case_report(case_id: str) -> dict[str, Any]:
    """Analyze a built-in demo case without an upload step."""
    path = _resolve_demo_case_path(case_id)
    report = analyze_vcf_path(path, batch_size=50_000)
    report["source_file"] = path.name
    return report


async def _analyze_uploaded_file(file: UploadFile) -> dict[str, Any]:
    """Validate an uploaded VCF, store it temporarily, and run analysis."""
    tmp_path: str | None = None
    source_filename = file.filename or "uploaded.vcf"

    try:
        _validate_filename(file.filename)

        suffix = ".vcf.gz" if source_filename.lower().endswith(".gz") else ".vcf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name

        total_bytes = 0
        while chunk := await file.read(8 * 1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise _error(400, f"File exceeds maximum upload size of {MAX_UPLOAD_BYTES // (1024**3)} GB.")
            tmp.write(chunk)
        tmp.close()

        if total_bytes == 0:
            raise _error(400, "Uploaded file is empty (0 bytes).")

        report = analyze_vcf_path(tmp_path, batch_size=50_000)
        report["source_file"] = source_filename
        return report

    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# Endpoint
# ============================================================================

@app.get("/")
async def frontend_index() -> FileResponse:
    """Serve the vanilla frontend application."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/demo-cases")
async def demo_cases() -> JSONResponse:
    """List built-in demo VCF cases for the frontend quick-run workflow."""
    return JSONResponse(content={"demo_cases": _demo_case_catalog()})


@app.get("/health")
async def health() -> JSONResponse:
    """Cheap liveness probe: the process is up and serving requests."""
    return JSONResponse(
        content={
            "status": "ok",
            "service": "vcf-variant-risk-analyzer",
        }
    )


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe: confirms critical in-memory artifacts are loaded."""
    checks = _readiness_checks()
    is_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
        },
    )


@app.get("/model-info")
async def model_info() -> JSONResponse:
    """Return metadata about the currently loaded model."""
    return JSONResponse(content=_build_model_info())


@app.get("/reports/{report_id}")
async def get_saved_report(report_id: str) -> JSONResponse:
    """Return a previously saved report payload by id."""
    try:
        return JSONResponse(
            content={
                "status": "completed",
                **_load_saved_report(report_id),
            }
        )
    except _APIError as exc:
        logger.warning("Saved report error: %s", exc.body["message"])
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except Exception as exc:
        logger.exception("Unexpected error during /reports/{report_id}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred while loading the saved report.",
                "details": str(exc),
            },
        )


@app.get("/reports/{report_id}/download")
async def download_saved_report(report_id: str):
    """Download a previously saved report as a JSON file attachment."""
    try:
        _validate_report_id(report_id)
        file_path = _report_file_path(report_id)
        if not os.path.exists(file_path):
            raise _error(404, "Report not found.", details=f"report_id={report_id}")
        return FileResponse(
            path=file_path,
            media_type="application/json",
            filename=f"{report_id}.json",
        )
    except _APIError as exc:
        logger.warning("Saved report error: %s", exc.body["message"])
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except Exception as exc:
        logger.exception("Unexpected error during /reports/{report_id}/download")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred while downloading the saved report.",
                "details": str(exc),
            },
        )


@app.post("/demo-cases/{case_id}/analyze")
async def analyze_demo_case(case_id: str) -> JSONResponse:
    """Run analysis for a built-in demo VCF case."""
    try:
        report = _run_demo_case_report(case_id)
        return JSONResponse(
            content={
                "status": "completed",
                **report,
            }
        )
    except _APIError as exc:
        logger.warning("Demo case error: %s", exc.body["message"])
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except Exception as exc:
        logger.exception("Unexpected error during /demo-cases/{case_id}/analyze")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred during demo analysis.",
                "details": str(exc),
            },
        )


@app.post("/analyze")
async def analyze_vcf(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a VCF file upload, scan variants, and return top risks.

    Validation order:
        1. File extension (.vcf or .vcf.gz)
        2. Non-empty body
        3. VCF format (parser checks for #CHROM header)

    Returns:
        JSON with scan summary, top_risks (above threshold), and
        top_candidates (best raw scores even if they stay below threshold).
    """
    try:
        report = await _analyze_uploaded_file(file)

        return JSONResponse(
            content={
                "status": "completed",
                **report,
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


@app.post("/demo-cases/{case_id}/report")
async def analyze_demo_case_report(case_id: str) -> JSONResponse:
    """Run report generation for a built-in demo VCF case."""
    try:
        report = _run_demo_case_report(case_id)
        clinical_report = _build_clinical_report(report["source_file"], report)
        storage = _save_report_payload(
            source_filename=report["source_file"],
            analysis_report=report,
            clinical_report=clinical_report,
        )

        return JSONResponse(
            content={
                "status": "completed",
                "source_file": report["source_file"],
                "report_storage": storage,
                "clinical_report": clinical_report,
            }
        )
    except _APIError as exc:
        logger.warning("Demo case error: %s", exc.body["message"])
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except Exception as exc:
        logger.exception("Unexpected error during /demo-cases/{case_id}/report")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred during demo report generation.",
                "details": str(exc),
            },
        )


@app.post("/analyze/report")
async def analyze_vcf_report(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a VCF upload and return a compact report-oriented response."""
    try:
        report = await _analyze_uploaded_file(file)
        clinical_report = _build_clinical_report(report["source_file"], report)
        storage = _save_report_payload(
            source_filename=report["source_file"],
            analysis_report=report,
            clinical_report=clinical_report,
        )

        return JSONResponse(
            content={
                "status": "completed",
                "source_file": report["source_file"],
                "report_storage": storage,
                "clinical_report": clinical_report,
            }
        )

    except _APIError as exc:
        logger.warning("Validation error: %s", exc.body["message"])
        return JSONResponse(status_code=exc.status_code, content=exc.body)

    except ValueError as exc:
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
        logger.exception("Unexpected error during /analyze/report")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred during report generation.",
                "details": str(exc),
            },
        )


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
