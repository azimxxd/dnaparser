const state = {
  lastPayload: null,
  lastMode: "analyze",
  lastSourceLabel: "",
};

const readyPill = document.querySelector("#ready-pill");
const serviceStatus = document.querySelector("#service-status");
const modelType = document.querySelector("#model-type");
const rocAuc = document.querySelector("#roc-auc");
const trainingRows = document.querySelector("#training-rows");
const supportedScope = document.querySelector("#supported-scope");
const scopeNotes = document.querySelector("#scope-notes");
const featureCount = document.querySelector("#feature-count");
const featurePreview = document.querySelector("#feature-preview");
const thresholdValue = document.querySelector("#threshold-value");
const diseaseMapSize = document.querySelector("#disease-map-size");

const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#vcf-file");
const selectedFile = document.querySelector("#selected-file");
const dropzone = document.querySelector(".dropzone");
const demoGrid = document.querySelector("#demo-grid");
const demoTemplate = document.querySelector("#demo-card-template");

const requestState = document.querySelector("#request-state");
const resultActions = document.querySelector("#result-actions");
const downloadJsonButton = document.querySelector("#download-json");
const downloadSavedReportLink = document.querySelector("#download-saved-report");
const copyReportIdButton = document.querySelector("#copy-report-id");
const emptyState = document.querySelector("#empty-state");
const resultsRoot = document.querySelector("#results-root");

function currentMode() {
  return document.querySelector('input[name="run-mode"]:checked')?.value || "analyze";
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US").format(Number(value));
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(6);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setRequestState(message, tone = "info") {
  requestState.hidden = false;
  requestState.className = `request-state ${tone}`;
  requestState.textContent = message;
}

function clearRequestState() {
  requestState.hidden = true;
  requestState.textContent = "";
  requestState.className = "request-state";
}

function setReadyChip(status, label) {
  readyPill.textContent = label;
  readyPill.className = "status-chip";
  readyPill.classList.add(status);
}

function badge(label, variant = "") {
  const klass = String(variant || "").replace(/\s+/g, "_").toLowerCase();
  return `<span class="badge ${klass}">${escapeHtml(label)}</span>`;
}

function renderTable(title, rows, columns, emptyText) {
  return `
    <section class="table-card">
      <div class="panel-headline">
        <div class="headline-main">
          <h3 class="panel-title">${escapeHtml(title)}</h3>
        </div>
      </div>
      ${
        rows.length
          ? `
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    ${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}
                  </tr>
                </thead>
                <tbody>
                  ${rows.map((row) => `
                    <tr>
                      ${columns.map((column) => `<td>${column.render(row)}</td>`).join("")}
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          `
          : `<p class="muted">${escapeHtml(emptyText)}</p>`
      }
    </section>
  `;
}

function renderOverviewStats(cards) {
  return `
    <section class="stats-grid">
      ${cards.map((card) => `
        <article class="stat-card">
          <span class="stat-label">${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
          <p class="summary-copy">${escapeHtml(card.note)}</p>
        </article>
      `).join("")}
    </section>
  `;
}

function renderCompactDetails(title, items, emptyText) {
  return `
    <section class="panel">
      <h3 class="panel-title">${escapeHtml(title)}</h3>
      ${
        items.length
          ? `<ul class="compact-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
          : `<p class="muted">${escapeHtml(emptyText)}</p>`
      }
    </section>
  `;
}

function renderAnalyzePayload(payload) {
  const summary = payload.analysis_summary;
  const guidance = payload.follow_up_guidance;
  const topRisks = (payload.top_risks || []).slice(0, 5);
  const topCandidates = (payload.top_candidates || []).slice(0, 5);

  const statCards = [
    {
      label: "Alert level",
      value: summary.overall_alert_level,
      note: "Current file-level signal after scoring.",
    },
    {
      label: "Dangerous variants",
      value: String(summary.dangerous_variants_found),
      note: "Variants above the active threshold.",
    },
    {
      label: "Max raw score",
      value: formatScore(summary.max_raw_score),
      note: "Highest pathogenic-like score in the file.",
    },
    {
      label: "Top gene",
      value: summary.top_genes?.[0]?.label || "n/a",
      note: "Most repeated gene in the strongest findings.",
    },
  ];

  const findingsColumns = [
    {
      label: "Variant",
      render: (item) => `<code>${escapeHtml(`${item.chromosome}:${item.position} ${item.mutation}`)}</code>`,
    },
    {
      label: "Score",
      render: (item) => escapeHtml(formatScore(item.risk_score)),
    },
    {
      label: "Gene",
      render: (item) => escapeHtml(item.gene || "n/a"),
    },
    {
      label: "Alert / confidence",
      render: (item) => `
        <div class="badge-row">
          ${badge(item.alert_level, item.alert_level)}
          ${badge(item.confidence_level || "n/a", "confidence")}
        </div>
      `,
    },
    {
      label: "Disease",
      render: (item) => escapeHtml(item.associated_disease || "n/a"),
    },
  ];

  return `
    <div class="results-stack">
      <section class="panel">
        <div class="panel-headline">
          <div class="headline-main">
            <h3 class="panel-title">${escapeHtml(summary.short_text)}</h3>
            <p class="summary-copy">
              File scanned: ${escapeHtml(String(summary.total_variants_scanned))} variants.
              Dangerous found: ${escapeHtml(String(summary.dangerous_variants_found))}.
              Threshold used: ${escapeHtml(String(payload.threshold_used))}.
            </p>
          </div>
          <div class="badge-row">
            ${badge(summary.overall_alert_level, summary.overall_alert_level)}
            ${badge(`urgency: ${guidance.urgency}`, guidance.urgency)}
          </div>
        </div>
      </section>

      ${renderOverviewStats(statCards)}

      <div class="compact-grid">
        ${renderCompactDetails(
          "Next steps",
          (guidance.recommended_next_steps || []).slice(0, 5),
          "No next-step hints available."
        )}
        ${renderCompactDetails(
          "Known signal summary",
          [
            `Reference hits in top risks: ${summary.reference_hits_in_top_risks}`,
            `Reference hits in top candidates: ${summary.reference_hits_in_top_candidates}`,
            `Top consequence: ${summary.top_molecular_consequences?.[0]?.label || "n/a"}`,
            `Dangerous rate: ${summary.dangerous_variant_rate}`,
          ],
          "No summary available."
        )}
      </div>

      ${renderTable(
        "Top high-risk findings",
        topRisks,
        findingsColumns,
        "No variants crossed the current threshold."
      )}

      ${renderTable(
        "Top candidates",
        topCandidates,
        findingsColumns,
        "No candidates available."
      )}

      <section class="panel">
        <h3 class="panel-title">Details on demand</h3>
        <details>
          <summary>Open raw JSON</summary>
          <pre class="raw-json">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </details>
      </section>
    </div>
  `;
}

function renderReportPayload(payload) {
  const report = payload.clinical_report;
  const storage = payload.report_storage || null;
  const findings = (report.main_findings || []).slice(0, 5);
  const statCards = [
    {
      label: "Alert level",
      value: report.overview.overall_alert_level,
      note: "Compact file-level result for this report.",
    },
    {
      label: "Urgency",
      value: report.overview.urgency,
      note: "How strongly this file should be reviewed next.",
    },
    {
      label: "Dangerous found",
      value: String(report.overview.dangerous_variants_found),
      note: "Variants above the current threshold.",
    },
    {
      label: "Saved report",
      value: storage?.report_id || "n/a",
      note: "Server-side JSON report id.",
    },
  ];

  const reportColumns = [
    {
      label: "Variant",
      render: (item) => `<code>${escapeHtml(item.variant)}</code>`,
    },
    {
      label: "Gene",
      render: (item) => escapeHtml(item.gene || "n/a"),
    },
    {
      label: "Score",
      render: (item) => escapeHtml(formatScore(item.risk_score)),
    },
    {
      label: "Alert / confidence",
      render: (item) => `
        <div class="badge-row">
          ${badge(item.alert_level, item.alert_level)}
          ${badge(item.confidence_level || "n/a", "confidence")}
          ${item.reference_model_conflict ? badge("conflict", "medium") : ""}
        </div>
      `,
    },
    {
      label: "Disease",
      render: (item) => escapeHtml(item.associated_disease || "n/a"),
    },
  ];

  return `
    <div class="results-stack">
      <section class="panel">
        <div class="panel-headline">
          <div class="headline-main">
            <h3 class="panel-title">${escapeHtml(report.headline)}</h3>
            <p class="summary-copy">
              ${escapeHtml((report.plain_language_summary || []).slice(0, 2).join(" "))}
            </p>
          </div>
          <div class="badge-row">
            ${badge(report.overview.overall_alert_level, report.overview.overall_alert_level)}
            ${badge(`urgency: ${report.overview.urgency}`, report.overview.urgency)}
          </div>
        </div>
      </section>

      ${renderOverviewStats(statCards)}

      <div class="compact-grid">
        ${renderCompactDetails(
          "Report summary",
          [
            report.interpretation?.what_stands_out || "n/a",
            report.interpretation?.what_this_means || "n/a",
            `Focus gene: ${report.interpretation?.focus_genes?.[0]?.label || "n/a"}`,
            `Focus consequence: ${report.interpretation?.focus_molecular_consequences?.[0]?.label || "n/a"}`,
          ],
          "No report summary available."
        )}
        ${renderCompactDetails(
          "Suggested next steps",
          (report.recommended_follow_up?.recommended_next_steps || []).slice(0, 5),
          "No follow-up steps available."
        )}
      </div>

      ${renderTable(
        "Main findings",
        findings,
        reportColumns,
        "No main findings in this report."
      )}

      <section class="panel">
        <h3 class="panel-title">Files and saved outputs</h3>
        <ul class="download-list">
          <li>Use "Download current JSON" for the exact payload from this browser run.</li>
          <li>Use "Download saved report" for the JSON file stored on the server.</li>
          <li>Use "Copy report id" if you want to reopen the report later through the API.</li>
        </ul>
        ${storage ? `<p class="download-note">Saved report id: <code>${escapeHtml(storage.report_id)}</code></p>` : ""}
        <details>
          <summary>Open raw JSON</summary>
          <pre class="raw-json">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </details>
      </section>
    </div>
  `;
}

function renderPayload(payload, mode) {
  return mode === "report" ? renderReportPayload(payload) : renderAnalyzePayload(payload);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.message || "Request failed");
    error.payload = data;
    throw error;
  }
  return data;
}

function downloadObjectAsJson(fileName, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    setRequestState("Report id copied.", "success");
  } catch {
    setRequestState("Could not copy report id automatically.", "error");
  }
}

function configureResultActions(payload, mode) {
  resultActions.hidden = false;

  const suggestedName =
    mode === "report"
      ? `${payload.source_file || "report"}-clinical-report.json`
      : `${payload.source_file || "analysis"}-analysis.json`;

  downloadJsonButton.onclick = () => downloadObjectAsJson(suggestedName, payload);

  const storage = payload.report_storage || null;
  if (storage?.report_id) {
    downloadSavedReportLink.hidden = false;
    downloadSavedReportLink.href = `/reports/${storage.report_id}/download`;
    downloadSavedReportLink.download = `${storage.report_id}.json`;

    copyReportIdButton.hidden = false;
    copyReportIdButton.onclick = () => copyText(storage.report_id);
  } else {
    downloadSavedReportLink.hidden = true;
    downloadSavedReportLink.removeAttribute("href");
    copyReportIdButton.hidden = true;
    copyReportIdButton.onclick = null;
  }
}

function showPayload(payload, mode, sourceLabel) {
  state.lastPayload = payload;
  state.lastMode = mode;
  state.lastSourceLabel = sourceLabel;
  emptyState.hidden = true;
  resultsRoot.hidden = false;
  resultsRoot.innerHTML = renderPayload(payload, mode);
  configureResultActions(payload, mode);
}

async function loadStatus() {
  try {
    const [ready, info] = await Promise.all([
      fetchJson("/ready"),
      fetchJson("/model-info"),
    ]);

    setReadyChip(ready.status === "ready" ? "ready" : "warn", ready.status === "ready" ? "Backend ready" : "Backend not ready");
    serviceStatus.textContent = ready.status;
    modelType.textContent = info.model_type || "n/a";
    rocAuc.textContent = info.training_summary?.test_metrics?.roc_auc?.toFixed(6) || "n/a";
    trainingRows.textContent = formatNumber(info.training_summary?.rows_after_filtering);
    supportedScope.textContent = (info.supported_scope?.variant_types || []).join(", ") || "n/a";
    thresholdValue.textContent = String(info.dangerous_threshold ?? "n/a");
    featureCount.textContent = `${info.feature_columns?.length || 0}`;
    diseaseMapSize.textContent = formatNumber(info.disease_map_entries);
    featurePreview.textContent = `Features: ${(info.feature_columns || []).slice(0, 8).join(", ")}`;
    scopeNotes.textContent = (info.supported_scope?.notes || []).join(" ");
  } catch (error) {
    setReadyChip("error", "Backend error");
    serviceStatus.textContent = "error";
    modelType.textContent = "n/a";
    rocAuc.textContent = "n/a";
    trainingRows.textContent = "n/a";
    supportedScope.textContent = "n/a";
    thresholdValue.textContent = "n/a";
    featureCount.textContent = "n/a";
    diseaseMapSize.textContent = "n/a";
    featurePreview.textContent = "Could not load model info.";
    scopeNotes.textContent = "Check whether the backend is running.";
  }
}

function renderDemoCases(items) {
  demoGrid.innerHTML = "";
  if (!items.length) {
    demoGrid.innerHTML = `<p class="muted">No demo cases found.</p>`;
    return;
  }

  for (const item of items) {
    const node = demoTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".demo-card-name").textContent = item.label;
    node.querySelector(".demo-card-hint").textContent = item.description;
    node.addEventListener("click", () => runDemoCase(item.id, item.filename));
    demoGrid.appendChild(node);
  }
}

async function loadDemoCases() {
  try {
    const data = await fetchJson("/demo-cases");
    renderDemoCases(data.demo_cases || []);
  } catch {
    demoGrid.innerHTML = `<p class="muted">Could not load demo cases.</p>`;
  }
}

async function runDemoCase(caseId, filename) {
  const mode = currentMode();
  setRequestState(`Running ${caseId} in ${mode} mode...`, "info");
  try {
    const payload = await fetchJson(`/demo-cases/${caseId}/${mode}`, { method: "POST" });
    showPayload(payload, mode, filename || caseId);
    setRequestState(`Demo case ${caseId} finished successfully.`, "success");
  } catch (error) {
    setRequestState(error.payload?.message || error.message, "error");
  }
}

async function runUpload(file) {
  const mode = currentMode();
  const formData = new FormData();
  formData.append("file", file);
  setRequestState(`Uploading ${file.name} in ${mode} mode...`, "info");

  try {
    const payload = await fetchJson(mode === "report" ? "/analyze/report" : "/analyze", {
      method: "POST",
      body: formData,
    });
    showPayload(payload, mode, file.name);
    setRequestState(`File ${file.name} finished successfully.`, "success");
  } catch (error) {
    setRequestState(error.payload?.message || error.message, "error");
  }
}

function attachUploadEvents() {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    selectedFile.textContent = file ? file.name : "No file selected";
  });

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) {
      setRequestState("Choose a VCF file first.", "error");
      return;
    }
    await runUpload(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) {
      return;
    }
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    selectedFile.textContent = file.name;
  });
}

async function init() {
  clearRequestState();
  attachUploadEvents();
  await Promise.all([loadStatus(), loadDemoCases()]);
}

init();
