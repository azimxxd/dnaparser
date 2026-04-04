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
from fastapi.responses import HTMLResponse, JSONResponse

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
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Lightweight health check for Docker and monitoring."""
    return {"status": "healthy", "model_features": len(FEATURE_COLUMNS)}


_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VCF Variant Risk Analyzer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#080c14;color:#e2e8f0;min-height:100vh;overflow-x:hidden}
.bg{position:fixed;inset:0;z-index:0;overflow:hidden}
.bg::before,.bg::after{content:'';position:absolute;border-radius:50%;filter:blur(120px);opacity:.35;animation:float 12s ease-in-out infinite}
.bg::before{width:600px;height:600px;background:radial-gradient(circle,#06b6d4,transparent 70%);top:-10%;left:-5%;animation-delay:0s}
.bg::after{width:500px;height:500px;background:radial-gradient(circle,#8b5cf6,transparent 70%);bottom:-10%;right:-5%;animation-delay:-6s}
@keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(40px,30px) scale(1.1)}}
.container{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:2.5rem 1.5rem}
header{text-align:center;margin-bottom:2.5rem;animation:fadeIn .8s ease-out}
@keyframes fadeIn{from{opacity:0;transform:translateY(-16px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
h1{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#06b6d4,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.4rem}
.subtitle{color:rgba(255,255,255,.5);font-size:1rem}
.glass{background:rgba(255,255,255,.04);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:2rem;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,.3);animation:slideUp .6s ease-out both}
.upload-zone{border:2px dashed rgba(255,255,255,.12);border-radius:12px;padding:3rem 2rem;text-align:center;cursor:pointer;transition:all .3s ease;position:relative}
.upload-zone:hover,.upload-zone.dragover{border-color:#06b6d4;background:rgba(6,182,212,.05);box-shadow:0 0 30px rgba(6,182,212,.1)}
.upload-zone svg{opacity:.4;transition:opacity .3s}.upload-zone:hover svg{opacity:.7}
.upload-zone p{color:rgba(255,255,255,.45);margin-top:.8rem;font-size:.92rem}
.upload-zone p strong{color:rgba(255,255,255,.7)}
.file-info{margin-top:.8rem;color:#06b6d4;font-weight:600;font-size:.9rem}
.file-info .size{color:rgba(255,255,255,.4);font-weight:400;margin-left:.5rem}
.btn-wrap{text-align:center;margin-top:1.2rem}
.btn{background:linear-gradient(135deg,#06b6d4,#8b5cf6);background-size:200% 200%;color:#fff;border:none;padding:.8rem 2.2rem;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;transition:all .3s;animation:gradShift 4s ease infinite}
@keyframes gradShift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(6,182,212,.3)}
.btn:disabled{opacity:.3;cursor:not-allowed;transform:none;box-shadow:none}
.progress{text-align:center;margin-top:1.2rem;display:none}
.progress .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#06b6d4;margin-right:8px;animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
.progress .status-text{color:rgba(255,255,255,.6);font-size:.9rem}
.error{color:#ef4444;text-align:center;margin-top:1rem;display:none}
.hidden{display:none!important}
.results{animation:slideUp .6s ease-out both;animation-delay:.1s}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:1.5rem}
.stat{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.2rem;text-align:center;transition:transform .2s}
.stat:hover{transform:translateY(-2px)}
.stat .label{color:rgba(255,255,255,.4);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem}
.stat .value{font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#06b6d4,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.chart-wrap{margin-bottom:1.5rem;display:flex;justify-content:center}
.chart-wrap canvas{max-height:280px}
.tbl-wrap{overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,.06)}
table{width:100%;border-collapse:collapse;font-size:.85rem}
thead{position:sticky;top:0;z-index:2}
th{text-align:left;padding:.75rem 1rem;background:rgba(255,255,255,.04);color:rgba(255,255,255,.45);font-weight:600;font-size:.73rem;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid rgba(255,255,255,.06)}
td{padding:.7rem 1rem;border-bottom:1px solid rgba(255,255,255,.03);transition:all .2s}
tr{animation:fadeIn .4s ease-out both}
tr:nth-child(even) td{background:rgba(255,255,255,.015)}
tr:hover td{background:rgba(6,182,212,.06);border-left-color:#06b6d4}
tr td:first-child{border-left:3px solid transparent;transition:border-color .2s}
tr:hover td:first-child{border-left-color:#06b6d4}
.pill{display:inline-block;padding:.2rem .65rem;border-radius:20px;font-weight:700;font-size:.8rem}
.pill-red{background:rgba(239,68,68,.15);color:#ef4444}
.pill-amber{background:rgba(245,158,11,.15);color:#f59e0b}
.disease-tag{display:inline-block;background:rgba(139,92,246,.12);color:#a78bfa;padding:.15rem .6rem;border-radius:6px;font-size:.78rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.actions{display:flex;gap:1rem;margin-top:1.2rem;justify-content:center}
.btn-sm{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:rgba(255,255,255,.7);padding:.5rem 1.2rem;border-radius:8px;font-size:.85rem;cursor:pointer;transition:all .2s}
.btn-sm:hover{background:rgba(255,255,255,.1);color:#fff}
footer{text-align:center;margin-top:3rem;color:rgba(255,255,255,.2);font-size:.78rem}
footer span{color:rgba(255,255,255,.35)}
input[type=file]{display:none}
</style>
</head>
<body>
<div class="bg"></div>
<div class="container">
<header>
  <h1>&#x1f9ec; VCF Variant Risk Analyzer</h1>
  <p class="subtitle">Upload a VCF file to scan for pathogenic mutations using XGBoost</p>
</header>

<div class="glass" id="uploadCard">
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <svg width="52" height="52" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 16V4m0 0L8 8m4-4 4 4"/><path d="M20 16.7V19a2 2 0 01-2 2H6a2 2 0 01-2-2v-2.3"/></svg>
    <p>Click or drag &amp; drop a <strong>.vcf</strong> or <strong>.vcf.gz</strong> file</p>
    <div class="file-info hidden" id="fileInfo"><span id="fileName"></span><span class="size" id="fileSize"></span></div>
  </div>
  <input type="file" id="fileInput" accept=".vcf,.gz">
  <div class="btn-wrap"><button class="btn" id="analyzeBtn" disabled>Analyze Variants</button></div>
  <div class="progress" id="progress"><span class="dot"></span><span class="status-text" id="statusText">Uploading...</span></div>
  <div class="error" id="errorMsg"></div>
</div>

<div class="hidden" id="results">
  <div class="glass results">
    <div class="stats">
      <div class="stat"><div class="label">Variants Scanned</div><div class="value" id="totalScanned">0</div></div>
      <div class="stat"><div class="label">High-Risk Found</div><div class="value" id="risksFound">0</div></div>
      <div class="stat"><div class="label">Processing Time</div><div class="value" id="timeVal">—</div></div>
      <div class="stat"><div class="label">Status</div><div class="value" id="statusVal">—</div></div>
    </div>
  </div>
  <div class="glass results" style="animation-delay:.2s">
    <h3 style="color:rgba(255,255,255,.5);font-size:.85rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem">Chromosome Distribution</h3>
    <div class="chart-wrap"><canvas id="chromChart"></canvas></div>
  </div>
  <div class="glass results" style="animation-delay:.3s">
    <h3 style="color:rgba(255,255,255,.5);font-size:.85rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem">Top Dangerous Mutations</h3>
    <div class="tbl-wrap">
      <table><thead><tr><th>#</th><th>Chr</th><th>Position</th><th>Mutation</th><th>Risk</th><th>Disease</th></tr></thead><tbody id="tbody"></tbody></table>
    </div>
    <div class="actions">
      <button class="btn-sm" id="csvBtn">Download CSV</button>
    </div>
  </div>
</div>

<footer>Powered by <span>XGBoost + ClinVar</span></footer>
</div>

<script>
const $=id=>document.getElementById(id);
const dropZone=$("dropZone"),fileInput=$("fileInput"),analyzeBtn=$("analyzeBtn"),
      progress=$("progress"),statusText=$("statusText"),errorEl=$("errorMsg"),
      resultsEl=$("results"),fileInfoEl=$("fileInfo");
let selectedFile=null,chartInstance=null,lastData=null;

function fmtSize(b){if(b<1024)return b+" B";if(b<1048576)return(b/1024).toFixed(1)+" KB";return(b/1048576).toFixed(1)+" MB"}

function selectFile(f){
  selectedFile=f;
  $("fileName").textContent=f.name;
  $("fileSize").textContent=fmtSize(f.size);
  fileInfoEl.classList.remove("hidden");
  analyzeBtn.disabled=false;
}

function countUp(el,target,dur=1200){
  const start=performance.now();
  const fmt=n=>n.toLocaleString();
  (function step(now){
    const p=Math.min((now-start)/dur,1);
    const ease=1-Math.pow(1-p,3);
    el.textContent=fmt(Math.floor(target*ease));
    if(p<1)requestAnimationFrame(step);
  })(start);
}

fileInput.addEventListener("change",e=>{if(e.target.files[0])selectFile(e.target.files[0])});
dropZone.addEventListener("dragover",e=>{e.preventDefault();dropZone.classList.add("dragover")});
dropZone.addEventListener("dragleave",()=>dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop",e=>{e.preventDefault();dropZone.classList.remove("dragover");if(e.dataTransfer.files[0])selectFile(e.dataTransfer.files[0])});

analyzeBtn.addEventListener("click",async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true;
  progress.style.display="block";errorEl.style.display="none";
  resultsEl.classList.add("hidden");
  statusText.textContent="Uploading...";
  const t0=performance.now();
  const fd=new FormData();fd.append("file",selectedFile);
  try{
    statusText.textContent="Analyzing variants...";
    const r=await fetch("/analyze",{method:"POST",body:fd});
    const d=await r.json();
    if(d.status==="error"){errorEl.textContent=d.message;errorEl.style.display="block";return}
    statusText.textContent="Building report...";
    lastData=d;
    const elapsed=((performance.now()-t0)/1000).toFixed(1)+"s";
    await new Promise(r=>setTimeout(r,300));
    countUp($("totalScanned"),d.total_variants_scanned);
    countUp($("risksFound"),d.top_risks.length);
    $("timeVal").textContent=elapsed;
    $("statusVal").textContent=d.status;
    const tb=$("tbody");tb.innerHTML="";
    d.top_risks.forEach((v,i)=>{
      const cls=v.risk_score>.95?"pill pill-red":"pill pill-amber";
      const dis=v.associated_disease&&v.associated_disease!=="Novel/Unknown Pathology"
        ?`<span class="disease-tag" title="`+v.associated_disease+`">`+v.associated_disease+`</span>`
        :`<span style="color:rgba(255,255,255,.25)">Unknown</span>`;
      const tr=document.createElement("tr");
      tr.style.animationDelay=(i*30)+"ms";
      tr.innerHTML=`<td>${i+1}</td><td>${v.chromosome}</td><td>${v.position}</td><td>${v.mutation}</td><td><span class="${cls}">${v.risk_score.toFixed(4)}</span></td><td>${dis}</td>`;
      tb.appendChild(tr);
    });
    renderChart(d.top_risks);
    resultsEl.classList.remove("hidden");
    setTimeout(()=>resultsEl.scrollIntoView({behavior:"smooth",block:"start"}),100);
  }catch(e){errorEl.textContent="Network error: "+e.message;errorEl.style.display="block"}
  finally{progress.style.display="none";analyzeBtn.disabled=false}
});

function renderChart(risks){
  const counts={};
  risks.forEach(v=>{const c=v.chromosome;counts[c]=(counts[c]||0)+1});
  const labels=Object.keys(counts).sort((a,b)=>{
    const na=parseInt(a.replace(/\\D/g,"")),nb=parseInt(b.replace(/\\D/g,""));
    return(isNaN(na)?99:na)-(isNaN(nb)?99:nb);
  });
  const data=labels.map(l=>counts[l]);
  const colors=["#06b6d4","#8b5cf6","#f59e0b","#ef4444","#10b981","#ec4899","#3b82f6","#f97316","#14b8a6","#a855f7","#64748b","#22d3ee","#e879f9","#84cc16","#fb923c","#38bdf8","#c084fc","#facc15","#f87171","#4ade80","#818cf8","#fb7185","#2dd4bf","#a3e635"];
  if(chartInstance)chartInstance.destroy();
  chartInstance=new Chart($("chromChart"),{type:"doughnut",data:{labels,datasets:[{data,backgroundColor:colors.slice(0,labels.length),borderWidth:0,hoverOffset:8}]},options:{responsive:true,plugins:{legend:{position:"right",labels:{color:"rgba(255,255,255,.5)",font:{size:11},padding:8,usePointStyle:true,pointStyleWidth:8}}}}});
}

$("csvBtn").addEventListener("click",()=>{
  if(!lastData)return;
  let csv="Rank,Chromosome,Position,Mutation,Risk Score,Disease\\n";
  lastData.top_risks.forEach((v,i)=>{
    csv+=[i+1,v.chromosome,v.position,`"${v.mutation}"`,v.risk_score.toFixed(6),`"${(v.associated_disease||"Unknown").replace(/"/g,"''")}"`].join(",")+("\\n");
  });
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="vcf_analysis_results.csv";a.click();
});
</script>
</body>
</html>
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
