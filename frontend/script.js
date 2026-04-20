/* ================================================================
   NeuroScan AI — frontend logic
   ================================================================ */

const API_PREDICT = '/api/predict';
const API_STATUS  = '/api/status';

// DOM refs
const dropZone     = document.getElementById('drop-zone');
const dropInner    = document.getElementById('drop-inner');
const fileInput    = document.getElementById('file-input');
const previewState = document.getElementById('preview-state');
const previewImg   = document.getElementById('preview-img');
const previewName  = document.getElementById('preview-name');
const clearBtn     = document.getElementById('clear-btn');
const analyzeBtn   = document.getElementById('analyze-btn');
const btnText      = analyzeBtn.querySelector('.btn-text');
const btnSpinner   = analyzeBtn.querySelector('.btn-spinner');

const emptyState   = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');
const errorState   = document.getElementById('error-state');
const errorMsg     = document.getElementById('error-message');

const statusBadge  = document.getElementById('model-status-badge');
const statusLabel  = statusBadge.querySelector('.status-label');

const toast        = document.getElementById('status-toast');
const toastMsg     = document.getElementById('toast-msg');
const toastClose   = document.getElementById('toast-close');

let selectedFile   = null;

/* ----------------------------------------------------------------
   Model status check on load
---------------------------------------------------------------- */
async function checkStatus() {
  try {
    const res = await fetch(API_STATUS);
    const data = await res.json();
    const p1 = data.phase1.loaded;
    const p2 = data.phase2.loaded;

    if (p1 && p2) {
      setStatus('ready', 'Both models ready');
    } else if (p1 || p2) {
      const which = p1 ? 'Phase 1 only' : 'Phase 2 only';
      setStatus('partial', which + ' loaded');
      showToast(`Only ${which} model loaded. Some results may be unavailable.`);
    } else {
      setStatus('error', 'Models not loaded');
      showToast('No trained models found. Run training first, then restart the server.');
    }
  } catch {
    setStatus('error', 'Server unreachable');
  }
}

function setStatus(type, label) {
  statusBadge.className = `status-badge status-${type}`;
  statusLabel.textContent = label;
}

/* ----------------------------------------------------------------
   File selection
---------------------------------------------------------------- */
dropZone.addEventListener('click', (e) => {
  if (e.target === clearBtn || clearBtn.contains(e.target)) return;
  // Label already opens the file dialog via its `for` attribute — don't double-trigger
  const label = dropZone.querySelector('label[for="file-input"]');
  if (label && (e.target === label || label.contains(e.target))) return;
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// Drag-and-drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});

clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  resetUpload();
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    dropInner.style.display = 'none';
    previewState.style.display = 'block';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);

  // Reset results when a new file is chosen
  showEmpty();
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewName.textContent = '';
  dropInner.style.display = 'flex';
  previewState.style.display = 'none';
  analyzeBtn.disabled = true;
  showEmpty();
}

/* ----------------------------------------------------------------
   Analysis
---------------------------------------------------------------- */
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  setLoadingState(true);

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch(API_PREDICT, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || 'Analysis failed. Please try again.');
      return;
    }

    renderResults(data);
  } catch (err) {
    showError('Could not reach the server. Make sure it is running.');
  } finally {
    setLoadingState(false);
  }
});

function setLoadingState(loading) {
  analyzeBtn.disabled = loading;
  btnText.style.display  = loading ? 'none'   : 'inline';
  btnSpinner.style.display = loading ? 'flex' : 'none';
}

/* ----------------------------------------------------------------
   Render results
---------------------------------------------------------------- */
function renderResults(data) {
  const { detection, classification, summary } = data;

  if (!summary) {
    showError('Received an unexpected response from the server.');
    return;
  }

  showResults();

  const tumorPresent = summary.tumor_present;
  const detConf = summary.detection_confidence ?? 0;
  const tumorType = summary.tumor_type ?? null;
  const typeConf = summary.type_confidence ?? null;

  renderVerdict(tumorPresent, detConf);

  if (classification && classification.class_probabilities) {
    renderClassification(tumorType, typeConf, classification.class_probabilities);
    document.getElementById('type-section').style.display = 'block';
  } else {
    document.getElementById('type-section').style.display = 'none';
  }

  // Notices for partial model availability
  const notice = document.getElementById('phase-notice');
  const notices = [];
  if (data.detection_error)     notices.push('Detection model not loaded: ' + data.detection_error);
  if (data.classification_error) notices.push('Classification model not loaded: ' + data.classification_error);

  if (notices.length) {
    notice.style.display = 'block';
    notice.textContent = '⚠ ' + notices.join(' | ');
  } else {
    notice.style.display = 'none';
  }
}

function renderVerdict(tumorPresent, confidence) {
  const card     = document.getElementById('verdict-card');
  const iconWrap = document.getElementById('verdict-icon-wrap');
  const label    = document.getElementById('verdict-label');
  const conf     = document.getElementById('verdict-confidence');
  const ringFill = document.getElementById('ring-fill');
  const ringVal  = document.getElementById('ring-value');

  card.classList.remove('tumor', 'no-tumor');
  card.classList.add(tumorPresent ? 'tumor' : 'no-tumor');

  iconWrap.textContent = tumorPresent ? '🔴' : '✅';
  label.textContent = tumorPresent ? 'Tumor Detected' : 'No Tumor Found';
  conf.textContent  = `Detection confidence: ${confidence.toFixed(1)}%`;
  ringVal.textContent = `${Math.round(confidence)}%`;

  // Animate ring
  const circumference = 263.9;
  const offset = circumference - (confidence / 100) * circumference;
  // Trigger animation on next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ringFill.style.strokeDashoffset = offset;
    });
  });
}

function renderClassification(tumorType, typeConf, classProbs) {
  const badge = document.getElementById('type-badge');
  const conf  = document.getElementById('type-conf');
  const bars  = document.getElementById('prob-bars');

  const displayName = tumorType ? formatClassName(tumorType) : '—';
  badge.textContent = displayName;
  conf.textContent  = typeConf !== null ? `${typeConf.toFixed(1)}% confidence` : '';

  // Sort classes by probability descending
  const sorted = Object.entries(classProbs).sort((a, b) => b[1] - a[1]);

  bars.innerHTML = '';
  sorted.forEach(([cls, pct], i) => {
    const isTop = i === 0;
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.style.animationDelay = `${i * 0.07}s`;
    row.innerHTML = `
      <span class="prob-name">${formatClassName(cls)}</span>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top-class' : 'other-class'}" data-width="${pct}"></div>
      </div>
      <span class="prob-pct">${pct.toFixed(1)}%</span>
    `;
    bars.appendChild(row);
  });

  // Animate bars after DOM insertion
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bars.querySelectorAll('.prob-fill').forEach((fill) => {
        fill.style.width = fill.dataset.width + '%';
      });
    });
  });
}

function formatClassName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ----------------------------------------------------------------
   Panel state helpers
---------------------------------------------------------------- */
function showEmpty() {
  emptyState.style.display    = 'flex';
  resultsContent.style.display = 'none';
  errorState.style.display    = 'none';
}

function showResults() {
  emptyState.style.display    = 'none';
  resultsContent.style.display = 'block';
  errorState.style.display    = 'none';
}

function showError(msg) {
  emptyState.style.display    = 'none';
  resultsContent.style.display = 'none';
  errorState.style.display    = 'flex';
  errorMsg.textContent        = msg;
}

/* ----------------------------------------------------------------
   Toast
---------------------------------------------------------------- */
function showToast(msg) {
  toastMsg.textContent = msg;
  toast.style.display  = 'flex';
}

toastClose.addEventListener('click', () => {
  toast.style.display = 'none';
});

/* ----------------------------------------------------------------
   Init
---------------------------------------------------------------- */
checkStatus();
