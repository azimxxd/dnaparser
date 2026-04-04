"""
FastAPI REST API for VCF variant risk analysis.

Accepts a VCF file upload, runs each variant through an XGBoost classifier,
and returns the top 50 most dangerous mutations sorted by risk score.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import tempfile
import urllib.error
import urllib.request
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

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

# Gemini settings for AI interpretation.
GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

# In-memory mock auth storage (resets on server restart by design).
MOCK_USERS: dict[str, dict[str, str]] = {}
MOCK_TOKENS: dict[str, str] = {}


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


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class VariantRisk(BaseModel):
    chromosome: str
    position: str | int
    mutation: str
    risk_score: float
    associated_disease: str | None = None


class AIReviewRequest(BaseModel):
    total_variants_scanned: int = Field(ge=0)
    top_risks: list[VariantRisk] = Field(default_factory=list)
    language: str = "ru"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str) -> str:
    # Mock auth hash only. Replace with bcrypt/argon2 for production.
    salted = f"clinical-lens-mock::{password}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def _is_valid_email(email: str) -> bool:
    email = email.strip()
    return "@" in email and "." in email.rsplit("@", 1)[-1]


def _build_ai_prompt(req: AIReviewRequest) -> str:
    """Build a focused clinical-style summary prompt for Gemini."""
    top = req.top_risks[:12]
    if top:
        lines = []
        for i, v in enumerate(top, start=1):
            lines.append(
                f"{i}. chr={v.chromosome}, pos={v.position}, mut={v.mutation}, "
                f"risk={v.risk_score:.4f}, disease={v.associated_disease or 'Unknown'}"
            )
        top_block = "\n".join(lines)
    else:
        top_block = "No high-risk variants were detected."

    language = "Russian" if req.language.lower().startswith("ru") else "English"
    return (
        "You are a genomic analysis assistant. "
        "Provide a concise clinical-style interpretation from model output.\n\n"
        f"Response language: {language}\n"
        "Output format:\n"
        "1) One short overall assessment paragraph.\n"
        "2) 3-5 bullet points with key risk signals.\n"
        "3) A caution section explicitly saying this is not a diagnosis and requires clinician review.\n"
        "Avoid fabricating genes or diseases not present in the data.\n\n"
        f"Total variants scanned: {req.total_variants_scanned}\n"
        f"Top high-risk variants (up to 12):\n{top_block}\n"
    )


def _fallback_ai_review(req: AIReviewRequest) -> str:
    """Fallback summary when Google API is unavailable."""
    total = req.total_variants_scanned
    flagged = len(req.top_risks)
    critical = sum(1 for v in req.top_risks if v.risk_score > 0.95)
    strongest = req.top_risks[0] if req.top_risks else None

    unique_diseases = sorted(
        {
            v.associated_disease
            for v in req.top_risks
            if v.associated_disease and v.associated_disease != "Novel/Unknown Pathology"
        }
    )
    diseases_text = ", ".join(unique_diseases[:5]) if unique_diseases else "No known disease associations detected"

    if req.language.lower().startswith("ru"):
        first_line = f"Проанализировано вариантов: {total}. Высокорисковых находок: {flagged}."
        strongest_line = (
            f"Наиболее значимый вариант: chr{strongest.chromosome} позиция {strongest.position}, "
            f"мутация {strongest.mutation}, риск {strongest.risk_score:.4f}."
            if strongest
            else "Критически значимых мутаций в top_risks не обнаружено."
        )
        return (
            f"{first_line}\n\n"
            f"- Критических вариантов (риск > 0.95): {critical}\n"
            f"- Известные ассоциации заболеваний: {diseases_text}\n"
            f"- {strongest_line}\n\n"
            "Важно: это автоматическая интерпретация модели и не является медицинским диагнозом. "
            "Для клинических решений требуется валидация специалистом."
        )

    first_line = f"Variants scanned: {total}. High-risk findings: {flagged}."
    strongest_line = (
        f"Top variant: chr{strongest.chromosome} position {strongest.position}, "
        f"mutation {strongest.mutation}, risk {strongest.risk_score:.4f}."
        if strongest
        else "No critical variants were detected in top_risks."
    )
    return (
        f"{first_line}\n\n"
        f"- Critical variants (risk > 0.95): {critical}\n"
        f"- Known disease associations: {diseases_text}\n"
        f"- {strongest_line}\n\n"
        "Important: this is an automated model interpretation and not a diagnosis. "
        "Clinical validation is required before medical decisions."
    )


def _generate_review_with_google(prompt: str) -> str:
    """Call Gemini REST API and return plain text response."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    model = os.getenv("GOOGLE_GEMINI_MODEL", GOOGLE_GEMINI_MODEL)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 700,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        logger.error("Gemini HTTP error: %s %s", exc.code, raw[:500])
        raise RuntimeError(f"Gemini API HTTP error: {exc.code}") from exc
    except urllib.error.URLError as exc:
        logger.error("Gemini network error: %s", exc)
        raise RuntimeError("Gemini API network error.") from exc

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    return text


def _generate_ai_review(req: AIReviewRequest) -> tuple[str, str]:
    """Generate AI review and return (text, source)."""
    prompt = _build_ai_prompt(req)
    try:
        text = _generate_review_with_google(prompt)
        return text, "google"
    except Exception as exc:
        logger.warning("Falling back to local AI summary: %s", exc)
        return _fallback_ai_review(req), "fallback"


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
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Lightweight health check for Docker and monitoring."""
    return {"status": "healthy", "model_features": len(FEATURE_COLUMNS)}


@app.post("/auth/register")
async def register_user(payload: RegisterRequest) -> JSONResponse:
    """Mock registration endpoint (in-memory)."""
    full_name = payload.full_name.strip()
    email = _normalize_email(payload.email)
    password = payload.password

    if len(full_name) < 2:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Full name must be at least 2 characters."},
        )
    if not _is_valid_email(email):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Please provide a valid email address."},
        )
    if len(password) < 6:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Password must be at least 6 characters."},
        )
    if email in MOCK_USERS:
        return JSONResponse(
            status_code=409,
            content={"status": "error", "message": "User with this email already exists."},
        )

    MOCK_USERS[email] = {
        "full_name": full_name,
        "email": email,
        "password_hash": _hash_password(password),
    }
    token = secrets.token_urlsafe(24)
    MOCK_TOKENS[token] = email

    return JSONResponse(
        content={
            "status": "completed",
            "message": "Registration successful.",
            "token": token,
            "user": {"full_name": full_name, "email": email},
        }
    )


@app.post("/auth/login")
async def login_user(payload: LoginRequest) -> JSONResponse:
    """Mock login endpoint (in-memory)."""
    email = _normalize_email(payload.email)
    password_hash = _hash_password(payload.password)
    user = MOCK_USERS.get(email)

    if not user or user["password_hash"] != password_hash:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid email or password."},
        )

    token = secrets.token_urlsafe(24)
    MOCK_TOKENS[token] = email
    return JSONResponse(
        content={
            "status": "completed",
            "message": "Login successful.",
            "token": token,
            "user": {"full_name": user["full_name"], "email": user["email"]},
        }
    )


@app.post("/ai-review")
async def ai_review(payload: AIReviewRequest) -> JSONResponse:
    """Generate interpretation text from analyzed variant output."""
    review_text, source = await asyncio.to_thread(_generate_ai_review, payload)
    return JSONResponse(
        content={
            "status": "completed",
            "source": source,
            "model": os.getenv("GOOGLE_GEMINI_MODEL", GOOGLE_GEMINI_MODEL),
            "review": review_text,
        }
    )


_INDEX_HTML = """
<!DOCTYPE html>
<html class="light" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Clinical Lens VCF - Precision Genetic Analysis</title>
<link href="https://fonts.googleapis.com" rel="preconnect"/>
<link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<script>
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary-fixed-dim":"#b1c5ff","on-secondary-container":"#006e6e","error":"#ba1a1a",
        "on-primary-fixed-variant":"#00419e","primary-container":"#0047ab","secondary":"#006a6a",
        "on-primary-fixed":"#001946","on-secondary":"#ffffff","outline":"#737784",
        "tertiary-container":"#3c4e69","surface-container-high":"#e7e8e9","surface-variant":"#e1e3e4",
        "on-primary":"#ffffff","on-tertiary-fixed-variant":"#364862","on-tertiary-fixed":"#071c35",
        "surface-container-low":"#f3f4f5","on-error-container":"#93000a","on-tertiary":"#ffffff",
        "outline-variant":"#c3c6d5","inverse-surface":"#2e3132","inverse-primary":"#b1c5ff",
        "on-secondary-fixed-variant":"#004f4f","tertiary":"#253751","surface-bright":"#f8f9fa",
        "on-primary-container":"#a5bdff","secondary-fixed":"#93f2f2","background":"#f8f9fa",
        "surface-tint":"#2559bd","inverse-on-surface":"#f0f1f2","error-container":"#ffdad6",
        "tertiary-fixed-dim":"#b5c7e8","primary-fixed":"#dae2ff","on-error":"#ffffff",
        "surface-container":"#edeeef","surface":"#f8f9fa","surface-container-highest":"#e1e3e4",
        "on-surface":"#191c1d","on-surface-variant":"#434653","primary":"#00327d",
        "secondary-container":"#90efef","surface-container-lowest":"#ffffff","on-background":"#191c1d",
        "surface-dim":"#d9dadb","on-secondary-fixed":"#002020","on-tertiary-container":"#adbfdf",
        "tertiary-fixed":"#d4e3ff","secondary-fixed-dim":"#76d6d5"
      },
      borderRadius: {"DEFAULT":"0.25rem","lg":"0.5rem","xl":"0.75rem","full":"9999px"},
      fontFamily: {"headline":["Manrope"],"body":["Inter"],"label":["Inter"]}
    }
  }
}
</script>
<style>
body{font-family:'Inter',sans-serif}
h1,h2,h3,.font-headline{font-family:'Manrope',sans-serif}
.material-symbols-outlined{font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24}
.glass-panel{background:rgba(255,255,255,0.7);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px)}
.text-gradient{background:linear-gradient(135deg,#00327d 0%,#0047ab 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.page{display:none}.page.active{display:block}
.nav-link{cursor:pointer}
.nav-link.active-link{color:#1d4ed8;border-bottom:2px solid #1d4ed8;padding-bottom:4px}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeIn 0.5s ease-out both}
.upload-zone.dragover{border-color:#006a6a!important;background:rgba(0,106,106,0.04)!important}
@keyframes barber{from{background-position:0 0}to{background-position:1rem 0}}
.barber-anim{animation:barber 0.6s linear infinite}
</style>
</head>
<body class="bg-surface text-on-surface flex flex-col min-h-screen selection:bg-secondary-container selection:text-on-secondary-container">

<!-- ==================== NAVBAR ==================== -->
<nav class="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-xl shadow-sm font-['Manrope'] antialiased">
<div class="flex justify-between items-center h-16 px-6 max-w-[1920px] mx-auto w-full">
  <div class="flex items-center gap-8">
    <span class="text-xl font-bold tracking-tight text-blue-900 cursor-pointer" onclick="showPage('welcome')">Clinical Lens VCF</span>
    <div class="hidden md:flex items-center gap-6">
      <a class="nav-link text-sm font-semibold text-slate-500 hover:text-blue-900 transition-colors" data-page="welcome">Dashboard</a>
      <a class="nav-link text-sm font-medium text-slate-500 hover:text-blue-900 transition-colors" data-page="upload">Upload</a>
      <a class="nav-link text-sm font-medium text-slate-500 hover:text-blue-900 transition-colors" data-page="dashboard">Reports</a>
      <a class="nav-link text-sm font-medium text-slate-500 hover:text-blue-900 transition-colors" data-page="variants">Variants</a>
    </div>
  </div>
  <div class="flex items-center gap-4">
    <button id="notificationsBtn" class="p-2 text-slate-500 hover:bg-slate-50 rounded-lg transition-all active:scale-95"><span class="material-symbols-outlined">notifications</span></button>
    <button id="settingsBtn" class="p-2 text-slate-500 hover:bg-slate-50 rounded-lg transition-all active:scale-95"><span class="material-symbols-outlined">settings</span></button>
    <button id="userMenuBtn" class="h-8 w-8 rounded-full bg-primary-container flex items-center justify-center text-on-primary text-xs font-bold ring-2 ring-surface overflow-hidden">
      <span id="userIcon" class="material-symbols-outlined text-white text-lg">person</span>
      <span id="userInitial" class="hidden text-white text-xs font-bold"></span>
    </button>
  </div>
</div>
<div class="bg-slate-100 h-[1px] w-full absolute bottom-0 opacity-15"></div>
</nav>

<main class="pt-16 flex-grow">
<!-- ==================== WELCOME PAGE ==================== -->
<div id="page-welcome" class="page active">
<section class="relative overflow-hidden min-h-[calc(100vh-4rem)] flex items-center px-6 lg:px-12 bg-surface">
<div class="max-w-[1440px] mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
  <div class="lg:col-span-7 z-10 fade-in">
    <div class="inline-flex items-center px-3 py-1 rounded-full bg-secondary-container/30 text-on-secondary-container text-xs font-bold tracking-widest uppercase mb-6">
      <span class="material-symbols-outlined text-xs mr-1" style="font-variation-settings:'FILL' 1;">science</span>Clinical Intelligence
    </div>
    <h1 class="font-headline text-5xl lg:text-7xl font-extrabold tracking-tight text-on-surface leading-[1.1] mb-6">
      Scan and decode your VCF file with <span class="text-gradient">AI precision.</span>
    </h1>
    <p class="text-on-surface-variant text-lg lg:text-xl max-w-xl mb-10 leading-relaxed">
      Transform complex genomic data into actionable clinical insights. Our AI-driven engine parses Variant Call Format files with XGBoost-powered accuracy for research and diagnostic support.
    </p>
    <div class="flex flex-wrap gap-4">
      <button onclick="showPage('upload')" class="bg-gradient-to-br from-primary to-primary-container text-on-primary px-8 py-4 rounded-lg font-bold text-lg shadow-lg hover:brightness-110 active:scale-95 transition-all flex items-center gap-2">
        Get Started <span class="material-symbols-outlined">arrow_forward</span>
      </button>
      <button onclick="showPage('upload')" class="bg-surface-container-highest text-primary px-8 py-4 rounded-lg font-bold text-lg hover:bg-surface-variant active:scale-95 transition-all">
        Upload VCF
      </button>
    </div>
    <div class="mt-12 flex items-center gap-8 opacity-60 grayscale">
      <div class="flex flex-col"><span class="text-2xl font-bold">ClinVar</span><span class="text-xs uppercase tracking-tighter font-semibold">Database</span></div>
      <div class="w-px h-8 bg-outline-variant"></div>
      <div class="flex flex-col"><span class="text-2xl font-bold">XGBoost</span><span class="text-xs uppercase tracking-tighter font-semibold">ML Engine</span></div>
      <div class="w-px h-8 bg-outline-variant"></div>
      <div class="flex flex-col"><span class="text-2xl font-bold">50+</span><span class="text-xs uppercase tracking-tighter font-semibold">Top Mutations</span></div>
    </div>
  </div>
  <div class="lg:col-span-5 relative h-full flex justify-center lg:justify-end">
    <div class="relative w-full max-w-md aspect-square">
      <div class="absolute inset-0 bg-primary/5 rounded-[2rem] rotate-6 transform"></div>
      <div class="absolute inset-0 bg-secondary/5 rounded-[2rem] -rotate-3 transform"></div>
      <div class="relative w-full h-full rounded-[2rem] overflow-hidden shadow-2xl border-4 border-surface-container-lowest">
        <img alt="DNA Analysis Visual" class="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDKPOJ203cXaO8As7bCTEomMhN7hhNDoAKPbrM3jRd0dJ4GExYs1PWzxZMgArG7Rz0v2UivJHLok1NmfvpEH6M4FJSlXKNx0mFytR3K_qEpMQck2KDnhZnoIqCZSuO7zB6_EzAb7Av01sOLULTxCUpj8uxc0YjvL38BdGMm-vmSG4ZfrVdS5LTJ6U6pD5esv8_xPFbjymT25hE-9FzQSdUWvJSCq2yyNu5FN1WpXQNG59nWU0XI2Vxlh0HvE1qCHQWW07x5vPDHN3k"/>
        <div class="absolute bottom-6 left-6 right-6 glass-panel p-4 rounded-xl border border-white/20 shadow-xl">
          <div class="flex justify-between items-end mb-2">
            <div><p class="text-[10px] uppercase font-bold text-primary tracking-widest">Mutation Scan</p><p class="text-lg font-headline font-bold">BRCA1-v.042</p></div>
            <span class="text-secondary font-bold">98% Match</span>
          </div>
          <div class="w-full h-1 bg-surface-container-high rounded-full overflow-hidden"><div class="h-full bg-gradient-to-r from-secondary to-secondary-fixed w-3/4"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>
</section>

<!-- Features Bento Grid -->
<section class="py-24 px-6 lg:px-12 bg-surface-container-low">
<div class="max-w-[1440px] mx-auto">
  <div class="flex flex-col md:flex-row md:items-end justify-between mb-16 gap-6">
    <div class="max-w-2xl">
      <h2 class="font-headline text-3xl lg:text-4xl font-bold mb-4">Precision-First Engineering</h2>
      <p class="text-on-surface-variant leading-relaxed">We've built the Clinical Lens to handle the heavy lifting of genomic data processing, so you can focus on the patient narrative.</p>
    </div>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-12 gap-6">
    <div class="md:col-span-7 lg:col-span-8 bg-surface-container-lowest p-8 rounded-xl shadow-sm border border-outline-variant/10 flex flex-col justify-between group hover:shadow-md transition-shadow">
      <div>
        <div class="w-12 h-12 rounded-lg bg-primary-container/10 flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform"><span class="material-symbols-outlined text-3xl" style="font-variation-settings:'FILL' 1;">bolt</span></div>
        <h3 class="font-headline text-2xl font-bold mb-4 text-on-surface">Fast Scanning</h3>
        <p class="text-on-surface-variant text-base leading-relaxed max-w-lg">Experience rapid parsing for multi-megabyte VCF files. Our batch processing architecture ensures that even large datasets are analyzed efficiently with XGBoost inference.</p>
      </div>
      <div class="mt-12 flex items-center gap-4 text-xs font-bold text-primary tracking-widest uppercase">
        <span class="bg-primary/5 px-2 py-1 rounded">50K Batch Size</span><span class="bg-primary/5 px-2 py-1 rounded">VCF 4.2 Support</span>
      </div>
    </div>
    <div class="md:col-span-5 lg:col-span-4 bg-gradient-to-br from-tertiary to-on-tertiary-fixed-variant p-8 rounded-xl shadow-sm text-on-tertiary flex flex-col justify-between relative overflow-hidden group">
      <div class="absolute -right-8 -top-8 opacity-10 group-hover:scale-110 transition-transform"><span class="material-symbols-outlined text-[12rem]">clinical_notes</span></div>
      <div class="z-10">
        <div class="w-12 h-12 rounded-lg bg-white/10 flex items-center justify-center text-white mb-6"><span class="material-symbols-outlined text-3xl">psychology</span></div>
        <h3 class="font-headline text-2xl font-bold mb-4">Accurate AI Reports</h3>
        <p class="text-on-tertiary/80 text-base leading-relaxed">Beyond simple parsing, our XGBoost model cross-references variants against ClinVar to provide clinical significance scoring and disease association.</p>
      </div>
      <a class="mt-8 flex items-center gap-2 text-sm font-bold underline underline-offset-4 decoration-secondary hover:text-secondary transition-colors z-10 cursor-pointer" onclick="showPage('upload')">UPLOAD YOUR FILE</a>
    </div>
  </div>
</div>
</section>

<!-- CTA Section -->
<section class="py-24 px-6 lg:px-12 bg-surface">
<div class="max-w-[1000px] mx-auto text-center glass-panel rounded-3xl p-12 lg:p-20 border border-outline-variant/15">
  <h2 class="font-headline text-4xl lg:text-5xl font-extrabold text-primary mb-6">Ready for deeper genomic insights?</h2>
  <p class="text-on-surface-variant text-lg lg:text-xl max-w-2xl mx-auto mb-10">Upload your first VCF file today and receive a mutation risk report in seconds. Powered by XGBoost + ClinVar.</p>
  <div class="flex flex-col sm:flex-row justify-center items-center gap-6">
    <button onclick="showPage('upload')" class="w-full sm:w-auto bg-primary text-on-primary px-10 py-4 rounded-lg font-bold text-lg hover:bg-primary-container active:scale-95 transition-all">Upload VCF Now</button>
  </div>
</div>
</section>
</div>
<!-- ==================== UPLOAD PAGE ==================== -->
<div id="page-upload" class="page">
<div class="pt-8 pb-12 px-6 flex flex-col items-center justify-center fade-in">
<div class="max-w-4xl w-full">
  <div class="mb-10 text-center md:text-left flex flex-col md:flex-row md:items-end justify-between gap-6">
    <div>
      <span class="text-secondary font-semibold tracking-widest text-xs uppercase mb-2 block">Helix Precision Protocol</span>
      <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight text-primary font-headline">VCF Sequence Import</h1>
      <p class="text-on-surface-variant mt-3 max-w-xl text-lg">Initialize your genomic analysis. Our clinical lens engine performs real-time variant identification and risk scoring.</p>
    </div>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
    <div class="md:col-span-8 bg-surface-container-lowest rounded-xl p-1 shadow-sm overflow-hidden">
      <div id="dropZone" class="upload-zone border-2 border-dashed border-outline-variant/30 rounded-lg bg-surface-container-low p-8 md:p-12 flex flex-col items-center text-center transition-all hover:bg-surface-container hover:border-secondary/40 group cursor-pointer" onclick="document.getElementById('fileInput').click()">
        <div class="mb-6 relative">
          <div class="absolute inset-0 bg-secondary-container/20 blur-2xl rounded-full scale-150"></div>
          <span class="material-symbols-outlined text-6xl text-secondary relative group-hover:scale-110 transition-transform duration-300">upload_file</span>
        </div>
        <h3 class="text-xl font-bold text-on-surface mb-2">Drag & Drop Genomic Data</h3>
        <p class="text-on-surface-variant text-sm mb-4 max-w-sm">Support for .vcf and .vcf.gz files (up to 2GB).</p>
        <div id="fileInfo" class="hidden mb-4">
          <span id="fileName" class="text-primary font-bold"></span>
          <span id="fileSize" class="text-on-surface-variant text-sm ml-2"></span>
        </div>
        <button type="button" class="bg-gradient-to-br from-primary to-primary-container text-white px-8 py-3 rounded-lg font-semibold shadow-lg hover:shadow-primary-container/20 active:scale-95 transition-all flex items-center gap-2" onclick="event.stopPropagation();document.getElementById('fileInput').click()">
          <span class="material-symbols-outlined">add_circle</span>Browse VCF Files
        </button>
        <input type="file" id="fileInput" accept=".vcf,.gz" class="hidden"/>
      </div>
      <!-- Progress Section -->
      <div id="progressSection" class="p-8 border-t border-surface-variant/10 hidden">
        <div class="flex justify-between items-center mb-4">
          <div class="flex items-center gap-3">
            <span class="material-symbols-outlined text-secondary" style="font-variation-settings:'FILL' 1;">genetics</span>
            <div><h4 id="progressFileName" class="text-sm font-bold text-on-surface"></h4><p id="progressStatus" class="text-xs text-on-surface-variant">Uploading...</p></div>
          </div>
          <span id="progressPercent" class="text-secondary font-bold text-sm">0%</span>
        </div>
        <div class="h-3 w-full bg-surface-container-high rounded-full overflow-hidden mb-2">
          <div id="progressBar" class="h-full bg-gradient-to-r from-secondary to-secondary-fixed-dim transition-all duration-500 relative barber-anim" style="width:0%;background-image:linear-gradient(45deg,rgba(255,255,255,0.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,0.15) 50%,rgba(255,255,255,0.15) 75%,transparent 75%,transparent);background-size:1rem 1rem"></div>
        </div>
        <div class="flex justify-between text-[10px] uppercase tracking-tighter text-outline font-bold mt-3">
          <span id="step1" class="text-secondary">Uploading</span>
          <span id="step2" class="opacity-30">Parsing variants</span>
          <span id="step3" class="opacity-30">ML Scoring</span>
          <span id="step4" class="opacity-30">Final Report</span>
        </div>
      </div>
      <!-- Error -->
      <div id="errorMsg" class="p-6 hidden"><p class="text-error font-medium text-center"></p></div>
    </div>
    <!-- Side Info -->
    <div class="md:col-span-4 space-y-6">
      <div class="bg-primary p-6 rounded-xl text-on-primary shadow-xl relative overflow-hidden group">
        <div class="absolute -right-4 -bottom-4 opacity-10 transform group-hover:scale-110 transition-transform duration-700"><span class="material-symbols-outlined text-9xl">biotech</span></div>
        <h4 class="text-sm font-bold mb-2">Automated ClinVar Matching</h4>
        <p class="text-xs text-primary-fixed-dim leading-relaxed mb-4">Variants are cross-referenced against the ClinVar database to identify known pathogenic mutations and disease associations.</p>
        <div class="flex items-center gap-2"><span class="material-symbols-outlined text-secondary-fixed text-sm">check_circle</span><span class="text-xs font-bold">Enabled</span></div>
      </div>
      <div class="bg-surface-container-low rounded-xl p-6 border border-outline-variant/15">
        <div class="flex items-center gap-2 mb-3">
          <span class="material-symbols-outlined text-secondary text-sm">verified_user</span>
          <span class="text-[10px] font-bold uppercase text-secondary">XGBoost ML Model</span>
        </div>
        <p class="text-[11px] text-on-surface-variant leading-relaxed">Variants are scored using a pre-trained XGBoost classifier. Mutations with a risk score above 0.80 are flagged as high-risk pathogenic variants.</p>
      </div>
      <button id="analyzeBtn" disabled class="w-full bg-gradient-to-br from-primary to-primary-container text-white py-4 rounded-xl font-bold text-lg shadow-lg hover:brightness-110 active:scale-95 transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-2">
        <span class="material-symbols-outlined">play_arrow</span>Analyze Variants
      </button>
    </div>
  </div>
</div>
</div>
</div>
<!-- ==================== DASHBOARD PAGE ==================== -->
<div id="page-dashboard" class="page">
<div class="pt-8 pb-12 px-6 md:px-12 max-w-[1920px] mx-auto w-full fade-in">
  <header class="mb-8">
    <div class="flex items-center gap-2 text-xs font-medium uppercase tracking-widest text-outline mb-2">
      <span>Analysis</span><span class="material-symbols-outlined text-[10px]">chevron_right</span><span class="text-primary">Current Session</span>
    </div>
    <h1 class="font-headline font-extrabold text-4xl tracking-tight text-primary">Analysis Dashboard</h1>
  </header>
  <!-- KPI Cards -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
    <div class="bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/10 relative overflow-hidden">
      <div class="flex justify-between items-start mb-4">
        <span class="text-sm font-semibold text-outline uppercase tracking-wider">Variants Scanned</span>
        <span class="material-symbols-outlined text-primary bg-primary/5 p-2 rounded-lg">biotech</span>
      </div>
      <div id="kpiScanned" class="text-4xl font-headline font-bold text-on-surface">—</div>
      <div class="absolute -right-4 -bottom-4 opacity-5 pointer-events-none"><span class="material-symbols-outlined text-8xl">genetics</span></div>
    </div>
    <div class="bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/10">
      <div class="flex justify-between items-start mb-4">
        <span class="text-sm font-semibold text-outline uppercase tracking-wider">High-Risk Found</span>
        <span class="material-symbols-outlined text-error bg-error-container/20 p-2 rounded-lg">warning</span>
      </div>
      <div class="flex items-end gap-4">
        <div><div id="kpiHighRisk" class="text-4xl font-headline font-bold text-error">—</div><div class="text-xs font-medium text-outline">Pathogenic</div></div>
        <div class="h-10 w-[1px] bg-outline-variant/20 mb-1"></div>
        <div><div id="kpiTotal" class="text-4xl font-headline font-bold text-on-surface">—</div><div class="text-xs font-medium text-outline">Total Shown</div></div>
      </div>
    </div>
    <div class="bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/10">
      <div class="flex justify-between items-start mb-4">
        <span class="text-sm font-semibold text-outline uppercase tracking-wider">Processing Time</span>
        <span class="material-symbols-outlined text-secondary bg-secondary-container/20 p-2 rounded-lg">timer</span>
      </div>
      <div id="kpiTime" class="text-4xl font-headline font-bold text-on-surface">—</div>
      <div class="mt-4 flex items-center gap-2">
        <div class="flex -space-x-2">
          <div class="w-6 h-6 rounded-full bg-primary-container border-2 border-surface flex items-center justify-center text-[10px] text-white font-bold">AI</div>
          <div class="w-6 h-6 rounded-full bg-secondary-container border-2 border-surface flex items-center justify-center text-[10px] text-on-secondary-container font-bold">ML</div>
        </div>
        <span class="text-xs text-outline font-medium">XGBoost Engine</span>
      </div>
    </div>
  </div>
  <!-- Main Insights -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
    <div class="lg:col-span-2 flex flex-col gap-8">
      <!-- AI Summary -->
      <section class="bg-surface-container-lowest rounded-xl p-8 shadow-sm border border-outline-variant/10">
        <div class="flex items-center gap-3 mb-6">
          <div class="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-primary-container flex items-center justify-center">
            <span class="material-symbols-outlined text-white" style="font-variation-settings:'FILL' 1;">psychology</span>
          </div>
          <h2 class="font-headline font-bold text-2xl text-primary">AI Interpretation Summary</h2>
        </div>
        <div id="aiSummary" class="prose prose-slate max-w-none text-on-surface-variant font-body leading-relaxed space-y-4">
          <p class="text-lg font-medium text-on-surface">Upload a VCF file to generate your analysis report.</p>
        </div>
        <div class="mt-8 flex gap-4">
          <button onclick="showPage('variants')" class="px-6 py-2.5 rounded-lg bg-gradient-to-r from-primary to-primary-container text-white font-semibold text-sm shadow-md hover:opacity-90 transition-opacity flex items-center gap-2">
            <span class="material-symbols-outlined text-sm">table_chart</span>View All Variants
          </button>
          <button id="dashCsvBtn" class="px-6 py-2.5 rounded-lg bg-surface-container-highest text-primary font-semibold text-sm hover:bg-surface-variant transition-colors flex items-center gap-2">
            <span class="material-symbols-outlined text-sm">download</span>Export CSV
          </button>
        </div>
      </section>
      <!-- Charts Row -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-surface-container-low rounded-xl p-6 border border-outline-variant/10">
          <h3 class="text-sm font-bold text-on-surface mb-6 uppercase tracking-widest">Chromosome Distribution</h3>
          <div class="flex items-center justify-center py-4">
            <div class="relative w-48 h-48"><svg id="donutChart" class="w-full h-full" viewBox="0 0 100 100"><circle class="text-surface-container-high" cx="50" cy="50" fill="transparent" r="40" stroke="currentColor" stroke-width="12"></circle></svg>
              <div class="absolute inset-0 flex flex-col items-center justify-center"><span id="donutTotal" class="text-2xl font-bold">0</span><span class="text-[10px] text-outline uppercase">Total Hits</span></div>
            </div>
          </div>
          <div id="donutLegend" class="mt-4 flex flex-wrap justify-center gap-3 px-4"></div>
        </div>
        <div class="bg-surface-container-low rounded-xl p-6 border border-outline-variant/10">
          <h3 class="text-sm font-bold text-on-surface mb-6 uppercase tracking-widest">Risk Score Distribution</h3>
          <div id="riskBars" class="space-y-4"></div>
        </div>
      </div>
    </div>
    <!-- Sidebar -->
    <aside class="space-y-6">
      <div class="bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/10">
        <h3 class="text-xs font-bold text-on-surface mb-4 uppercase tracking-widest border-b border-outline-variant/10 pb-3">Analysis Metadata</h3>
        <div class="space-y-4">
          <div class="flex justify-between items-center"><span class="text-xs text-outline font-medium">Status</span><span id="metaStatus" class="text-xs px-2 py-1 bg-secondary-container rounded-full font-bold text-on-secondary-container">Pending</span></div>
          <div class="flex justify-between items-center"><span class="text-xs text-outline font-medium">Model</span><span class="text-sm font-bold text-on-surface">XGBoost</span></div>
          <div class="flex justify-between items-center"><span class="text-xs text-outline font-medium">Risk Threshold</span><span class="text-sm font-bold text-secondary">&gt; 0.80</span></div>
        </div>
      </div>
      <div class="bg-primary rounded-xl overflow-hidden relative group h-52 shadow-xl">
        <div class="absolute inset-0 bg-gradient-to-br from-primary to-primary-container"></div>
        <div class="absolute inset-0 flex flex-col justify-end p-6">
          <h4 class="text-white font-bold text-lg leading-tight">ClinVar Database</h4>
          <p class="text-primary-fixed-dim text-xs mt-2">Mutations matched against known pathogenic variants from the NCBI ClinVar database.</p>
          <button onclick="showPage('variants')" class="mt-4 w-full py-2 bg-white/20 backdrop-blur-md rounded-lg text-white text-xs font-bold hover:bg-white/30 transition-all">View Variants</button>
        </div>
      </div>
    </aside>
  </div>
</div>
</div>
<!-- ==================== VARIANTS TABLE PAGE ==================== -->
<div id="page-variants" class="page">
<div class="pt-8 pb-12 px-6 max-w-[1920px] mx-auto w-full fade-in">
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
    <div class="lg:col-span-2 space-y-2">
      <h1 class="text-3xl font-extrabold tracking-tight text-primary font-headline">Detailed Variant Analysis</h1>
      <p class="text-on-surface-variant max-w-2xl">Top pathogenic mutations sorted by risk score. Filter by chromosome or clinical significance.</p>
    </div>
    <div class="flex items-end justify-start lg:justify-end gap-3">
      <button id="tblCsvBtn" class="flex items-center gap-2 px-4 py-2 bg-surface-container-highest rounded-lg text-primary font-medium transition-all hover:bg-surface-variant">
        <span class="material-symbols-outlined text-[18px]">download</span>Export CSV
      </button>
      <button onclick="showPage('upload')" class="flex items-center gap-2 px-6 py-2 bg-gradient-to-br from-primary to-primary-container text-white rounded-lg font-medium shadow-md hover:shadow-lg transition-all">
        <span class="material-symbols-outlined text-[18px]">biotech</span>New Analysis
      </button>
    </div>
  </div>
  <!-- Filters -->
  <div class="glass-panel border-b border-white/20 rounded-xl p-6 mb-8 shadow-sm">
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="space-y-2">
        <label class="text-xs font-bold uppercase tracking-widest text-outline">Chromosome</label>
        <select id="filterChrom" class="w-full bg-white border-outline-variant/30 rounded-lg text-sm focus:ring-primary focus:border-primary"><option value="">All Chromosomes</option></select>
      </div>
      <div class="space-y-2">
        <label class="text-xs font-bold uppercase tracking-widest text-outline">Min Risk Score</label>
        <select id="filterRisk" class="w-full bg-white border-outline-variant/30 rounded-lg text-sm focus:ring-primary focus:border-primary">
          <option value="0">All (&gt; 0.80)</option><option value="0.90">&gt; 0.90</option><option value="0.95">&gt; 0.95</option><option value="0.99">&gt; 0.99</option>
        </select>
      </div>
      <div class="space-y-2">
        <label class="text-xs font-bold uppercase tracking-widest text-outline">Disease</label>
        <select id="filterDisease" class="w-full bg-white border-outline-variant/30 rounded-lg text-sm focus:ring-primary focus:border-primary">
          <option value="">All Diseases</option><option value="known">Known Only</option><option value="unknown">Unknown Only</option>
        </select>
      </div>
      <div class="space-y-2 flex items-end">
        <button id="applyFilters" class="w-full bg-primary text-white py-2.5 rounded-lg font-semibold text-sm hover:bg-primary-container transition-all flex items-center justify-center gap-2">
          <span class="material-symbols-outlined text-sm">filter_list</span>Apply Filters
        </button>
      </div>
    </div>
  </div>
  <!-- Data Table -->
  <div class="bg-surface-container-lowest rounded-xl shadow-sm overflow-hidden">
    <div class="overflow-x-auto">
      <table class="w-full text-left border-collapse">
        <thead><tr class="bg-surface-container-low border-b border-outline-variant/10">
          <th class="px-6 py-4 text-xs font-extrabold text-primary uppercase tracking-wider">#</th>
          <th class="px-6 py-4 text-xs font-extrabold text-primary uppercase tracking-wider">Chr</th>
          <th class="px-6 py-4 text-xs font-extrabold text-primary uppercase tracking-wider">Position</th>
          <th class="px-6 py-4 text-xs font-extrabold text-primary uppercase tracking-wider text-center">Mutation</th>
          <th class="px-6 py-4 text-xs font-extrabold text-primary uppercase tracking-wider">Risk Score</th>
          <th class="px-6 py-4 text-xs font-extrabold text-primary uppercase tracking-wider">Disease Association</th>
        </tr></thead>
        <tbody id="variantBody" class="divide-y divide-outline-variant/5"></tbody>
      </table>
    </div>
    <div id="variantPagination" class="px-6 py-4 flex items-center justify-between bg-surface-container-low/50"></div>
  </div>
  <!-- Info boxes -->
  <div class="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
    <div class="p-6 bg-tertiary text-white rounded-xl flex items-start gap-4">
      <div class="p-3 bg-white/10 rounded-lg"><span class="material-symbols-outlined text-secondary-fixed">clinical_notes</span></div>
      <div><h3 class="font-headline font-bold text-lg mb-1">Clinical Guidelines</h3><p class="text-sm opacity-80 leading-relaxed">Pathogenic markers are identified using an XGBoost classifier trained on ClinVar data. Variants with risk_score &gt; 0.80 are flagged as high-risk.</p></div>
    </div>
    <div class="p-6 bg-surface-container-low rounded-xl border border-outline-variant/10 flex items-start gap-4">
      <div class="p-3 bg-primary/10 rounded-lg"><span class="material-symbols-outlined text-primary">data_usage</span></div>
      <div><h3 class="font-headline font-bold text-lg text-primary mb-1">Filtering Summary</h3><p id="filterSummary" class="text-sm text-on-surface-variant leading-relaxed">Upload a VCF file to see variant analysis results.</p></div>
    </div>
  </div>
</div>
</div>
</div>
</main>

<!-- Toasts -->
<div id="toastContainer" class="fixed top-20 right-6 z-[80] space-y-2"></div>

<!-- Auth Modal -->
<div id="authModal" class="fixed inset-0 z-[70] bg-black/40 backdrop-blur-sm hidden items-center justify-center p-4">
  <div class="w-full max-w-md bg-white rounded-2xl shadow-2xl border border-outline-variant/20 p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-headline text-xl font-bold text-primary">Account Access (Mock)</h3>
      <button id="closeAuthModal" class="p-2 rounded-lg hover:bg-surface-container-high"><span class="material-symbols-outlined text-base">close</span></button>
    </div>
    <div class="grid grid-cols-2 gap-2 mb-4 bg-surface-container-low rounded-lg p-1">
      <button id="loginTabBtn" class="py-2 rounded-md text-sm font-semibold bg-white shadow-sm">Login</button>
      <button id="registerTabBtn" class="py-2 rounded-md text-sm font-semibold text-outline">Register</button>
    </div>
    <form id="authForm" class="space-y-3">
      <div id="fullNameRow" class="hidden">
        <label class="text-xs font-semibold text-outline">Full Name</label>
        <input id="authFullName" type="text" class="mt-1 w-full rounded-lg border-outline-variant/40 focus:border-primary focus:ring-primary" placeholder="Jane Doe"/>
      </div>
      <div>
        <label class="text-xs font-semibold text-outline">Email</label>
        <input id="authEmail" type="email" class="mt-1 w-full rounded-lg border-outline-variant/40 focus:border-primary focus:ring-primary" placeholder="jane@example.com" required/>
      </div>
      <div>
        <label class="text-xs font-semibold text-outline">Password</label>
        <input id="authPassword" type="password" class="mt-1 w-full rounded-lg border-outline-variant/40 focus:border-primary focus:ring-primary" placeholder="Minimum 6 chars" required/>
      </div>
      <button id="authSubmitBtn" type="submit" class="w-full py-2.5 rounded-lg bg-primary text-white font-semibold hover:bg-primary-container transition-colors">Login</button>
    </form>
    <p id="authMessage" class="text-sm mt-4 hidden"></p>
  </div>
</div>

<!-- Settings Modal -->
<div id="settingsModal" class="fixed inset-0 z-[70] bg-black/40 backdrop-blur-sm hidden items-center justify-center p-4">
  <div class="w-full max-w-md bg-white rounded-2xl shadow-2xl border border-outline-variant/20 p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="font-headline text-xl font-bold text-primary">Settings</h3>
      <button id="closeSettingsModal" class="p-2 rounded-lg hover:bg-surface-container-high"><span class="material-symbols-outlined text-base">close</span></button>
    </div>
    <div class="space-y-4">
      <div>
        <label class="text-xs font-semibold text-outline">AI Review Language</label>
        <select id="aiLanguage" class="mt-1 w-full rounded-lg border-outline-variant/40 focus:border-primary focus:ring-primary">
          <option value="ru">Russian</option>
          <option value="en">English</option>
        </select>
      </div>
      <p class="text-xs text-on-surface-variant">For real AI review, set <code>GOOGLE_API_KEY</code> in backend environment.</p>
      <button id="saveSettingsBtn" class="w-full py-2.5 rounded-lg bg-primary text-white font-semibold hover:bg-primary-container transition-colors">Save Preferences</button>
      <button id="logoutBtn" class="w-full py-2.5 rounded-lg bg-surface-container-highest text-primary font-semibold hover:bg-surface-variant transition-colors">Logout</button>
    </div>
  </div>
</div>

<!-- Info Modal -->
<div id="infoModal" class="fixed inset-0 z-[70] bg-black/40 backdrop-blur-sm hidden items-center justify-center p-4">
  <div class="w-full max-w-lg bg-white rounded-2xl shadow-2xl border border-outline-variant/20 p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 id="infoModalTitle" class="font-headline text-xl font-bold text-primary">Information</h3>
      <button id="closeInfoModal" class="p-2 rounded-lg hover:bg-surface-container-high"><span class="material-symbols-outlined text-base">close</span></button>
    </div>
    <div id="infoModalBody" class="text-sm leading-relaxed text-on-surface-variant space-y-2"></div>
  </div>
</div>

<!-- Footer -->
<footer class="w-full py-8 mt-auto bg-slate-50 border-t border-slate-200 font-['Inter'] text-xs tracking-wide">
<div class="flex flex-col md:flex-row justify-between items-center px-8 gap-4 max-w-[1920px] mx-auto w-full">
  <span class="text-slate-500">&copy; 2024 Clinical Lens VCF &mdash; Genetic Analysis Platform. Powered by XGBoost + ClinVar.</span>
  <div class="flex gap-6">
    <a class="footer-link text-slate-500 hover:text-blue-600 transition-opacity duration-200 hover:opacity-80" data-topic="docs" href="#">Documentation</a>
    <a class="footer-link text-slate-500 hover:text-blue-600 transition-opacity duration-200 hover:opacity-80" data-topic="privacy" href="#">Privacy Policy</a>
    <a class="footer-link text-slate-500 hover:text-blue-600 transition-opacity duration-200 hover:opacity-80" data-topic="support" href="#">Support</a>
  </div>
</div>
</footer>

<script>
/* === SPA Navigation === */
let currentPage='welcome';
function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  const el=document.getElementById('page-'+name);
  if(el){el.classList.add('active');currentPage=name;}
  document.querySelectorAll('.nav-link').forEach(l=>{
    l.classList.toggle('active-link',l.dataset.page===name);
    l.classList.toggle('text-slate-500',l.dataset.page!==name);
  });
  window.scrollTo({top:0,behavior:'smooth'});
}
document.querySelectorAll('.nav-link').forEach(l=>l.addEventListener('click',()=>showPage(l.dataset.page)));

/* === Generic Helpers === */
let selectedFile=null, lastData=null, analysisTime='';
const AUTH_STORAGE_KEY='clinical_lens_auth_v1';
const PREF_STORAGE_KEY='clinical_lens_prefs_v1';
let authMode='login';
let authState={token:'',user:null};
let prefs={aiLanguage:'ru'};

function fmtSize(b){if(b<1024)return b+' B';if(b<1048576)return(b/1024).toFixed(1)+' KB';if(b<1073741824)return(b/1048576).toFixed(1)+' MB';return(b/1073741824).toFixed(1)+' GB'}
function fmtNum(n){return Number(n||0).toLocaleString()}
function escapeHtml(s){
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function showToast(message,type){
  type=type||'info';
  const container=document.getElementById('toastContainer');
  const tone=type==='error'?'bg-error text-white':type==='success'?'bg-secondary text-white':'bg-primary text-white';
  const el=document.createElement('div');
  el.className='px-4 py-3 rounded-lg shadow-lg text-sm font-medium '+tone+' max-w-sm';
  el.textContent=message;
  container.appendChild(el);
  setTimeout(()=>{el.classList.add('opacity-0','translate-x-2')},2200);
  setTimeout(()=>el.remove(),2600);
}

function openModal(id){
  const el=document.getElementById(id);
  if(!el)return;
  el.classList.remove('hidden');
  el.classList.add('flex');
}

function closeModal(id){
  const el=document.getElementById(id);
  if(!el)return;
  el.classList.add('hidden');
  el.classList.remove('flex');
}

function setInfoModal(title,html){
  document.getElementById('infoModalTitle').textContent=title;
  document.getElementById('infoModalBody').innerHTML=html;
  openModal('infoModal');
}

function setAuthMode(mode){
  authMode=mode;
  const isRegister=mode==='register';
  document.getElementById('fullNameRow').classList.toggle('hidden',!isRegister);
  document.getElementById('authSubmitBtn').textContent=isRegister?'Create Account':'Login';
  document.getElementById('loginTabBtn').className=isRegister?'py-2 rounded-md text-sm font-semibold text-outline':'py-2 rounded-md text-sm font-semibold bg-white shadow-sm';
  document.getElementById('registerTabBtn').className=isRegister?'py-2 rounded-md text-sm font-semibold bg-white shadow-sm':'py-2 rounded-md text-sm font-semibold text-outline';
  const msg=document.getElementById('authMessage');
  msg.classList.add('hidden');
  msg.textContent='';
}

function syncUserBadge(){
  const icon=document.getElementById('userIcon');
  const initial=document.getElementById('userInitial');
  if(authState.user&&authState.user.full_name){
    const letter=authState.user.full_name.trim().charAt(0).toUpperCase()||'U';
    initial.textContent=letter;
    initial.classList.remove('hidden');
    icon.classList.add('hidden');
    document.getElementById('logoutBtn').disabled=false;
  }else{
    initial.classList.add('hidden');
    icon.classList.remove('hidden');
    document.getElementById('logoutBtn').disabled=true;
  }
}

function saveAuthState(){
  localStorage.setItem(AUTH_STORAGE_KEY,JSON.stringify(authState));
  syncUserBadge();
}

function loadState(){
  try{
    const raw=localStorage.getItem(AUTH_STORAGE_KEY);
    if(raw)authState=JSON.parse(raw);
  }catch(_){}
  try{
    const rawPrefs=localStorage.getItem(PREF_STORAGE_KEY);
    if(rawPrefs)prefs={...prefs,...JSON.parse(rawPrefs)};
  }catch(_){}
  document.getElementById('aiLanguage').value=prefs.aiLanguage||'ru';
  syncUserBadge();
}

function savePrefs(){
  prefs.aiLanguage=document.getElementById('aiLanguage').value||'ru';
  localStorage.setItem(PREF_STORAGE_KEY,JSON.stringify(prefs));
}

/* === Buttons Previously Inactive === */
document.getElementById('notificationsBtn').addEventListener('click',()=>{
  const notes=[];
  if(lastData){
    notes.push('Latest scan: '+lastData.top_risks.length+' high-risk variants from '+fmtNum(lastData.total_variants_scanned)+' scanned.');
  }else{
    notes.push('No analysis results yet. Upload a VCF file to start.');
  }
  notes.push(authState.user?'Signed in as '+authState.user.full_name+'.':'You are in guest mode.');
  setInfoModal(
    'Notifications',
    '<ul class="list-disc pl-5 space-y-1">'+notes.map(n=>'<li>'+escapeHtml(n)+'</li>').join('')+'</ul>'
  );
});

document.getElementById('settingsBtn').addEventListener('click',()=>openModal('settingsModal'));
document.getElementById('closeSettingsModal').addEventListener('click',()=>closeModal('settingsModal'));
document.getElementById('saveSettingsBtn').addEventListener('click',()=>{
  savePrefs();
  closeModal('settingsModal');
  showToast('Preferences saved.','success');
});
document.getElementById('logoutBtn').addEventListener('click',()=>{
  authState={token:'',user:null};
  saveAuthState();
  closeModal('settingsModal');
  showToast('Logged out.','info');
});

document.getElementById('userMenuBtn').addEventListener('click',()=>{
  if(authState.user){openModal('settingsModal');return;}
  openModal('authModal');
});
document.getElementById('closeAuthModal').addEventListener('click',()=>closeModal('authModal'));
document.getElementById('loginTabBtn').addEventListener('click',()=>setAuthMode('login'));
document.getElementById('registerTabBtn').addEventListener('click',()=>setAuthMode('register'));

document.getElementById('authForm').addEventListener('submit',async e=>{
  e.preventDefault();
  const msg=document.getElementById('authMessage');
  msg.classList.add('hidden');
  const endpoint=authMode==='register'?'/auth/register':'/auth/login';
  const payload={
    email:document.getElementById('authEmail').value.trim(),
    password:document.getElementById('authPassword').value
  };
  if(authMode==='register'){
    payload.full_name=document.getElementById('authFullName').value.trim();
  }
  try{
    const r=await fetch(endpoint,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const d=await r.json();
    if(!r.ok||d.status==='error'){
      msg.textContent=d.message||'Authentication failed.';
      msg.className='text-sm mt-4 text-error';
      msg.classList.remove('hidden');
      return;
    }
    authState={token:d.token,user:d.user};
    saveAuthState();
    closeModal('authModal');
    showToast(authMode==='register'?'Registration completed.':'Login successful.','success');
  }catch(err){
    msg.textContent='Network error: '+err.message;
    msg.className='text-sm mt-4 text-error';
    msg.classList.remove('hidden');
  }
});

document.querySelectorAll('.footer-link').forEach(link=>{
  link.addEventListener('click',e=>{
    e.preventDefault();
    const topic=link.dataset.topic;
    if(topic==='docs'){
      setInfoModal('Documentation','<p>Upload a <strong>.vcf</strong> or <strong>.vcf.gz</strong> file, run analysis, then review dashboard metrics and variant-level table.</p><p>Use Export CSV to download high-risk findings.</p>');
    }else if(topic==='privacy'){
      setInfoModal('Privacy Policy','<p>This demo stores mock auth data only in local browser storage and in-memory server data.</p><p>Do not upload real patient-identifiable data in demo mode.</p>');
    }else{
      setInfoModal('Support','<p>Support channel: open an issue in your repository or contact the project maintainer.</p><p>Attach server logs and sample input for faster troubleshooting.</p>');
    }
  });
});
document.getElementById('closeInfoModal').addEventListener('click',()=>closeModal('infoModal'));

['authModal','settingsModal','infoModal'].forEach(id=>{
  const modal=document.getElementById(id);
  modal.addEventListener('click',e=>{if(e.target===modal)closeModal(id)});
});

/* === File Upload === */
const dropZone=document.getElementById('dropZone'),fileInput=document.getElementById('fileInput'),
  analyzeBtn=document.getElementById('analyzeBtn'),progressSection=document.getElementById('progressSection'),
  errorEl=document.getElementById('errorMsg');

function selectFile(f){
  selectedFile=f;
  document.getElementById('fileName').textContent=f.name;
  document.getElementById('fileSize').textContent=fmtSize(f.size);
  document.getElementById('fileInfo').classList.remove('hidden');
  analyzeBtn.disabled=false;
}

fileInput.addEventListener('change',e=>{if(e.target.files[0])selectFile(e.target.files[0])});
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('dragover')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('dragover');if(e.dataTransfer.files[0])selectFile(e.dataTransfer.files[0])});

function setProgress(pct,status,activeStep){
  document.getElementById('progressBar').style.width=pct+'%';
  document.getElementById('progressPercent').textContent=pct+'%';
  document.getElementById('progressStatus').textContent=status;
  ['step1','step2','step3','step4'].forEach((s,i)=>{
    const el=document.getElementById(s);
    el.classList.toggle('text-secondary',i<=activeStep);
    el.classList.toggle('opacity-30',i>activeStep);
  });
}

analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true;
  progressSection.classList.remove('hidden');
  errorEl.classList.add('hidden');
  document.getElementById('progressFileName').textContent=selectedFile.name;
  setProgress(10,'Uploading file...',0);
  const t0=performance.now();
  const fd=new FormData();fd.append('file',selectedFile);
  try{
    setProgress(30,'Uploading...',0);
    const r=await fetch('/analyze',{method:'POST',body:fd});
    setProgress(60,'Analyzing variants...',1);
    const d=await r.json();
    if(d.status==='error'){
      errorEl.querySelector('p').textContent=d.message;
      errorEl.classList.remove('hidden');
      progressSection.classList.add('hidden');
      analyzeBtn.disabled=false;
      return;
    }
    setProgress(80,'Scoring mutations...',2);
    await new Promise(r=>setTimeout(r,400));
    setProgress(100,'Complete!',3);
    await new Promise(r=>setTimeout(r,300));
    lastData=d;
    analysisTime=((performance.now()-t0)/1000).toFixed(1)+'s';
    populateDashboard(d);
    populateVariants(d);
    progressSection.classList.add('hidden');
    showPage('dashboard');
    showToast('Analysis completed.','success');
  }catch(e){
    errorEl.querySelector('p').textContent='Network error: '+e.message;
    errorEl.classList.remove('hidden');
    progressSection.classList.add('hidden');
  }finally{analyzeBtn.disabled=false}
});

/* === Dashboard Population === */
function countUp(el,target,dur){
  dur=dur||1200;const start=performance.now();
  (function step(now){
    const p=Math.min((now-start)/dur,1),ease=1-Math.pow(1-p,3);
    el.textContent=fmtNum(Math.floor(target*ease));
    if(p<1)requestAnimationFrame(step);
  })(start);
}

const CHART_COLORS=['#00327d','#006a6a','#0047ab','#ba1a1a','#253751','#b1c5ff','#93f2f2','#3c4e69','#76d6d5','#b5c7e8','#737784','#c3c6d5','#e7e8e9','#f59e0b','#10b981','#8b5cf6','#ec4899','#3b82f6','#f97316','#14b8a6','#a855f7','#64748b','#22d3ee','#e879f9'];

function buildRuleSummaryHtml(d,highRisk){
  const topGene=d.top_risks[0];
  const diseases=[...new Set(d.top_risks.filter(v=>v.associated_disease&&v.associated_disease!=='Novel/Unknown Pathology').map(v=>v.associated_disease))];
  let html='<p class="text-lg font-medium text-on-surface">Genomic profiling identified <strong>'+d.top_risks.length+' high-risk mutations</strong> from '+fmtNum(d.total_variants_scanned)+' total variants scanned.</p>';
  if(topGene)html+='<p>The highest-risk variant was at chromosome <strong>'+escapeHtml(topGene.chromosome)+'</strong> position '+escapeHtml(topGene.position)+' ('+escapeHtml(topGene.mutation)+') with risk score <strong>'+topGene.risk_score.toFixed(4)+'</strong>.</p>';
  if(diseases.length>0){
    html+='<div class="bg-secondary-container/10 border-l-4 border-secondary p-4 rounded-r-lg my-6"><h4 class="text-secondary font-bold text-sm uppercase tracking-wide mb-1">Disease Associations</h4><p class="text-on-secondary-fixed-variant">'+diseases.slice(0,5).map(escapeHtml).join(', ')+(diseases.length>5?' and '+(diseases.length-5)+' more':'')+'</p></div>';
  }
  html+='<p>Variants with risk scores above 0.95 are considered critically pathogenic. '+highRisk+' variant(s) exceeded this threshold.</p>';
  return html;
}

function aiTextToHtml(text){
  const safe=escapeHtml(text||'');
  const blocks=safe.split(/\\n\\n+/).map(b=>b.trim()).filter(Boolean);
  if(blocks.length===0)return '<p>No AI review returned.</p>';
  return blocks.map(b=>'<p>'+b.replace(/\\n/g,'<br/>')+'</p>').join('');
}

async function requestAiReview(d){
  const aiBox=document.getElementById('aiSummary');
  aiBox.innerHTML='<p class="text-on-surface-variant">Generating AI interpretation from scanned variants...</p>';
  try{
    const r=await fetch('/ai-review',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        total_variants_scanned:d.total_variants_scanned,
        top_risks:d.top_risks,
        language:prefs.aiLanguage||'ru'
      })
    });
    const payload=await r.json();
    if(!r.ok||payload.status==='error'){
      throw new Error(payload.message||'AI review failed');
    }
    aiBox.innerHTML=aiTextToHtml(payload.review);
    if(payload.source==='fallback'){
      showToast('Google API недоступен, показан локальный fallback-отзыв.','info');
    }else{
      showToast('AI review generated via Google API.','success');
    }
  }catch(err){
    const highRisk=d.top_risks.filter(v=>v.risk_score>0.95).length;
    aiBox.innerHTML=buildRuleSummaryHtml(d,highRisk);
    showToast('AI review failed: '+err.message,'error');
  }
}

function populateDashboard(d){
  countUp(document.getElementById('kpiScanned'),d.total_variants_scanned);
  const highRisk=d.top_risks.filter(v=>v.risk_score>0.95).length;
  document.getElementById('kpiHighRisk').textContent=highRisk;
  document.getElementById('kpiTotal').textContent=d.top_risks.length;
  document.getElementById('kpiTime').textContent=analysisTime;
  document.getElementById('metaStatus').textContent=d.status==='completed'?'Completed':'Error';

  document.getElementById('aiSummary').innerHTML=buildRuleSummaryHtml(d,highRisk);
  requestAiReview(d);

  const chromCounts={};
  d.top_risks.forEach(v=>{chromCounts[v.chromosome]=(chromCounts[v.chromosome]||0)+1});
  const labels=Object.keys(chromCounts).sort((a,b)=>{const na=parseInt(a.replace(/\\D/g,'')),nb=parseInt(b.replace(/\\D/g,''));return(isNaN(na)?99:na)-(isNaN(nb)?99:nb)});
  const total=d.top_risks.length;
  document.getElementById('donutTotal').textContent=total;
  const svg=document.getElementById('donutChart');
  svg.innerHTML='<circle class="text-surface-container-high" cx="50" cy="50" fill="transparent" r="40" stroke="currentColor" stroke-width="12"></circle>';
  const circumference=2*Math.PI*40;
  let offset=0;
  labels.forEach((label,i)=>{
    const pct=chromCounts[label]/total;
    const dashLen=pct*circumference;
    const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx','50');circle.setAttribute('cy','50');circle.setAttribute('r','40');
    circle.setAttribute('fill','transparent');circle.setAttribute('stroke',CHART_COLORS[i%CHART_COLORS.length]);
    circle.setAttribute('stroke-width','12');circle.setAttribute('stroke-dasharray',dashLen+' '+(circumference-dashLen));
    circle.setAttribute('stroke-dashoffset',-offset);circle.setAttribute('transform','rotate(-90 50 50)');
    circle.setAttribute('stroke-linecap','round');
    svg.appendChild(circle);offset+=dashLen;
  });
  const legend=document.getElementById('donutLegend');legend.innerHTML='';
  labels.forEach((label,i)=>{
    legend.innerHTML+='<div class="flex items-center gap-1"><div class="w-2 h-2 rounded-full" style="background:'+CHART_COLORS[i%CHART_COLORS.length]+'"></div><span class="text-xs text-outline font-medium">'+escapeHtml(label)+' ('+chromCounts[label]+')</span></div>';
  });

  const riskBars=document.getElementById('riskBars');riskBars.innerHTML='';
  const brackets=[{label:'Critical (>0.99)',min:0.99},{label:'Very High (>0.95)',min:0.95},{label:'High (>0.90)',min:0.90},{label:'Elevated (>0.80)',min:0.80}];
  brackets.forEach(b=>{
    const count=d.top_risks.filter(v=>v.risk_score>b.min).length;
    const pct=total>0?Math.round(count/total*100):0;
    riskBars.innerHTML+='<div class="space-y-2"><div class="flex justify-between text-xs font-medium text-outline"><span>'+b.label+'</span><span>'+count+'</span></div><div class="h-2 w-full bg-surface-container-high rounded-full overflow-hidden"><div class="h-full bg-gradient-to-r from-secondary to-secondary-fixed rounded-full" style="width:'+pct+'%"></div></div></div>';
  });
}

/* === Variants Table === */
let allVariants=[], filteredVariants=[], currentTablePage=0;
const PAGE_SIZE=10;

function populateVariants(d){
  allVariants=d.top_risks;filteredVariants=[...allVariants];currentTablePage=0;
  const chroms=[...new Set(allVariants.map(v=>v.chromosome))].sort((a,b)=>{const na=parseInt(a.replace(/\\D/g,'')),nb=parseInt(b.replace(/\\D/g,''));return(isNaN(na)?99:na)-(isNaN(nb)?99:nb)});
  const sel=document.getElementById('filterChrom');sel.innerHTML='<option value="">All Chromosomes</option>';
  chroms.forEach(c=>sel.innerHTML+='<option value="'+c+'">'+c+'</option>');
  document.getElementById('filterSummary').textContent='Showing '+allVariants.length+' high-risk variants (risk > 0.80) from '+fmtNum(d.total_variants_scanned)+' total variants scanned.';
  renderTable();
}

function renderTable(){
  const tbody=document.getElementById('variantBody');tbody.innerHTML='';
  const start=currentTablePage*PAGE_SIZE;
  const pageData=filteredVariants.slice(start,start+PAGE_SIZE);
  pageData.forEach((v,i)=>{
    const idx=start+i+1;
    const ref=v.mutation.split(' -> ')[0]||'';const alt=v.mutation.split(' -> ')[1]||'';
    const riskPct=Math.round(v.risk_score*100);
    const disease=v.associated_disease&&v.associated_disease!=='Novel/Unknown Pathology'?'<span class="px-2 py-1 bg-secondary-container/30 text-on-secondary-container rounded text-xs font-medium max-w-[200px] truncate inline-block" title="'+escapeHtml(v.associated_disease)+'">'+escapeHtml(v.associated_disease)+'</span>':'<span class="text-outline text-xs">Unknown</span>';
    tbody.innerHTML+='<tr class="hover:bg-blue-50/30 transition-colors group"><td class="px-6 py-4 text-sm text-outline">'+idx+'</td><td class="px-6 py-4 font-semibold text-on-surface">'+escapeHtml(v.chromosome)+'</td><td class="px-6 py-4 font-mono text-sm text-on-surface-variant">'+Number(v.position).toLocaleString()+'</td><td class="px-6 py-4"><div class="flex items-center justify-center gap-3"><span class="px-2 py-1 bg-surface-container-high rounded font-mono text-xs">'+escapeHtml(ref)+'</span><span class="material-symbols-outlined text-outline text-xs">arrow_forward</span><span class="px-2 py-1 bg-secondary-container text-on-secondary-container rounded font-mono text-xs font-bold">'+escapeHtml(alt)+'</span></div></td><td class="px-6 py-4"><div class="flex items-center gap-3"><div class="w-20 h-1.5 bg-surface-container-high rounded-full overflow-hidden"><div class="h-full bg-gradient-to-r from-secondary to-secondary-fixed" style="width:'+riskPct+'%"></div></div><span class="text-xs font-bold">'+v.risk_score.toFixed(4)+'</span></div></td><td class="px-6 py-4">'+disease+'</td></tr>';
  });

  if(filteredVariants.length===0){
    tbody.innerHTML='<tr><td colspan="6" class="px-6 py-8 text-center text-sm text-outline">No variants match current filters.</td></tr>';
  }

  const totalPages=Math.ceil(filteredVariants.length/PAGE_SIZE);
  const pag=document.getElementById('variantPagination');
  if(filteredVariants.length===0){
    pag.innerHTML='<span class="text-xs text-on-surface-variant font-medium">Showing 0 variants</span>';
    return;
  }

  pag.innerHTML='<span class="text-xs text-on-surface-variant font-medium">Showing <span class="text-primary font-bold">'+(start+1)+'-'+Math.min(start+PAGE_SIZE,filteredVariants.length)+'</span> of '+filteredVariants.length+' variants</span>';
  if(totalPages>1){
    let btns='<div class="flex items-center gap-2"><button onclick="prevPage()" class="p-2 rounded-lg hover:bg-surface-container-high text-outline transition-colors'+(currentTablePage===0?' opacity-30 cursor-default':'')+'"><span class="material-symbols-outlined">chevron_left</span></button><div class="flex gap-1">';
    for(let p=0;p<totalPages&&p<7;p++){
      btns+='<button onclick="goPage('+p+')" class="w-8 h-8 flex items-center justify-center rounded-lg text-xs font-medium'+(p===currentTablePage?' bg-primary text-white font-bold':' hover:bg-surface-container-high')+'">'+(p+1)+'</button>';
    }
    if(totalPages>7)btns+='<span class="px-2">...</span><button onclick="goPage('+(totalPages-1)+')" class="w-8 h-8 flex items-center justify-center rounded-lg text-xs font-medium hover:bg-surface-container-high">'+totalPages+'</button>';
    btns+='</div><button onclick="nextPage()" class="p-2 rounded-lg hover:bg-surface-container-high text-outline transition-colors'+(currentTablePage>=totalPages-1?' opacity-30 cursor-default':'')+'"><span class="material-symbols-outlined">chevron_right</span></button></div>';
    pag.innerHTML+=btns;
  }
}
function prevPage(){if(currentTablePage>0){currentTablePage--;renderTable()}}
function nextPage(){if(currentTablePage<Math.ceil(filteredVariants.length/PAGE_SIZE)-1){currentTablePage++;renderTable()}}
function goPage(p){currentTablePage=p;renderTable()}

document.getElementById('applyFilters').addEventListener('click',()=>{
  const chrom=document.getElementById('filterChrom').value;
  const minRisk=parseFloat(document.getElementById('filterRisk').value)||0;
  const diseaseFilter=document.getElementById('filterDisease').value;
  filteredVariants=allVariants.filter(v=>{
    if(chrom&&v.chromosome!==chrom)return false;
    if(v.risk_score<=minRisk&&minRisk>0)return false;
    if(diseaseFilter==='known'&&(!v.associated_disease||v.associated_disease==='Novel/Unknown Pathology'))return false;
    if(diseaseFilter==='unknown'&&v.associated_disease&&v.associated_disease!=='Novel/Unknown Pathology')return false;
    return true;
  });
  currentTablePage=0;
  renderTable();
  showToast('Filters applied.','info');
});

/* === CSV Export === */
function downloadCsv(){
  if(!lastData||!lastData.top_risks||lastData.top_risks.length===0){
    showToast('No analysis data to export yet.','info');
    return;
  }
  let csv='Rank,Chromosome,Position,Mutation,Risk Score,Disease\\n';
  lastData.top_risks.forEach((v,i)=>{
    csv+=[i+1,v.chromosome,v.position,'"'+v.mutation+'"',v.risk_score.toFixed(6),'"'+(v.associated_disease||'Unknown').replace(/"/g,"''")+'"'].join(',')+'\\n';
  });
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='vcf_analysis_results.csv';a.click();
}
document.getElementById('dashCsvBtn').addEventListener('click',downloadCsv);
document.getElementById('tblCsvBtn').addEventListener('click',downloadCsv);

/* === Init === */
loadState();
setAuthMode('login');
showPage('welcome');
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page upload UI."""
    return _INDEX_HTML


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
