const ui = {
  exportButton: document.querySelector("#export-button"),
  saveStatus: document.querySelector("#save-status"),
  metricTotal: document.querySelector("#metric-total"),
  metricMismatches: document.querySelector("#metric-mismatches"),
  metricErrorRemaining: document.querySelector("#metric-error-remaining"),
  metricApproved: document.querySelector("#metric-approved"),
  metricCorrected: document.querySelector("#metric-corrected"),
  metricRejected: document.querySelector("#metric-rejected"),
  tabs: Array.from(document.querySelectorAll("[data-mode]")),
  leafFilter: document.querySelector("#leaf-filter"),
  searchInput: document.querySelector("#search-input"),
  previousButton: document.querySelector("#previous-button"),
  nextButton: document.querySelector("#next-button"),
  queuePosition: document.querySelector("#queue-position"),
  queueList: document.querySelector("#queue-list"),
  sampleImage: document.querySelector("#sample-image"),
  imageEmpty: document.querySelector("#image-empty"),
  sampleId: document.querySelector("#sample-id"),
  sampleTitle: document.querySelector("#sample-title"),
  sampleSource: document.querySelector("#sample-source"),
  sampleQuery: document.querySelector("#sample-query"),
  sampleLicense: document.querySelector("#sample-license"),
  sampleSplit: document.querySelector("#sample-split"),
  sampleModel: document.querySelector("#sample-model"),
  originalLabel: document.querySelector("#original-label"),
  originalLeaf: document.querySelector("#original-leaf"),
  predictedLabel: document.querySelector("#predicted-label"),
  predictedLeaf: document.querySelector("#predicted-leaf"),
  sampleStatus: document.querySelector("#sample-status"),
  top3List: document.querySelector("#top3-list"),
  labelSelect: document.querySelector("#label-select"),
  decisionNote: document.querySelector("#decision-note"),
  confirmOriginal: document.querySelector("#confirm-original"),
  saveCorrection: document.querySelector("#save-correction"),
  rejectSample: document.querySelector("#reject-sample"),
  clearDecision: document.querySelector("#clear-decision"),
  toast: document.querySelector("#toast"),
};

const state = {
  payload: null,
  mode: "errors",
  selectedId: null,
  filtered: [],
  busy: false,
  toastTimer: null,
};

function text(value, fallback = "-") {
  const normalized = String(value ?? "").trim();
  return normalized || fallback;
}

function scoreText(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}

function reviewStatusText(value) {
  return {
    model_mismatch: "모델 불일치",
    model_uncertain: "저신뢰",
    model_match: "모델 일치",
    auto_rejected: "자동 제외 후보",
    auto_approved: "자동 승인 후보",
    pending: "검수 대기",
    source_mapped: "소스 매핑",
  }[value] || value;
}

function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  ui.toast.textContent = message;
  ui.toast.classList.toggle("error", isError);
  ui.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    ui.toast.hidden = true;
  }, 5000);
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  if (state.payload?.mutation_token && options.method && options.method !== "GET") {
    headers.set("x-labeler-token", state.payload.mutation_token);
  }
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `요청 실패 (${response.status})`);
  }
  return payload;
}

function categoryGroups() {
  const groups = new Map();
  for (const category of state.payload.categories) {
    if (!groups.has(category.group_id)) {
      groups.set(category.group_id, { name: category.group_name, categories: [] });
    }
    groups.get(category.group_id).categories.push(category);
  }
  return groups;
}

function populateCategoryMenus() {
  ui.leafFilter.replaceChildren(new Option("전체 카테고리", ""));
  ui.labelSelect.replaceChildren();
  for (const group of categoryGroups().values()) {
    const filterGroup = document.createElement("optgroup");
    const labelGroup = document.createElement("optgroup");
    filterGroup.label = group.name;
    labelGroup.label = group.name;
    for (const category of group.categories) {
      const label = `${category.display_name} (${category.id})`;
      filterGroup.append(new Option(label, category.id));
      labelGroup.append(new Option(label, category.id));
    }
    ui.leafFilter.append(filterGroup);
    ui.labelSelect.append(labelGroup);
  }
}

function updateMetrics(summary) {
  ui.metricTotal.textContent = summary.total;
  ui.metricMismatches.textContent = summary.mismatches;
  ui.metricErrorRemaining.textContent = summary.mismatch_remaining;
  ui.metricApproved.textContent = summary.approved;
  ui.metricCorrected.textContent = summary.corrected;
  ui.metricRejected.textContent = summary.rejected;
  ui.saveStatus.textContent = `검수 ${summary.reviewed}/${summary.total}`;
}

function searchableText(sample) {
  return [
    sample.sample_id,
    sample.title,
    sample.query,
    sample.source,
    sample.original_leaf_id,
    sample.original_display_name,
    sample.predicted_leaf_id,
    sample.predicted_display_name,
  ]
    .filter(Boolean)
    .join(" ")
    .toLocaleLowerCase("ko");
}

function matchesMode(sample) {
  const reviewed = Boolean(sample.decision);
  if (state.mode === "errors") return sample.mismatch && !reviewed;
  if (state.mode === "needs_review") return sample.need_user_check && !reviewed;
  if (state.mode === "unreviewed") return !reviewed;
  if (state.mode === "reviewed") return reviewed;
  return true;
}

function applyFilters({ preserveSelection = true } = {}) {
  const leafId = ui.leafFilter.value;
  const query = ui.searchInput.value.trim().toLocaleLowerCase("ko");
  state.filtered = state.payload.samples.filter((sample) => {
    if (!matchesMode(sample)) return false;
    if (leafId && sample.original_leaf_id !== leafId) return false;
    return !query || searchableText(sample).includes(query);
  });
  if (!preserveSelection || !state.filtered.some((sample) => sample.sample_id === state.selectedId)) {
    state.selectedId = state.filtered[0]?.sample_id || null;
  }
  renderQueue();
  renderCurrent();
}

function queueItem(sample) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "queue-item";
  button.dataset.sampleId = sample.sample_id;
  button.setAttribute("aria-current", String(sample.sample_id === state.selectedId));
  button.addEventListener("click", () => {
    state.selectedId = sample.sample_id;
    renderQueue();
    renderCurrent();
  });

  const dot = document.createElement("span");
  dot.className = "queue-dot";
  if (sample.decision) dot.classList.add("reviewed");
  else if (sample.mismatch) dot.classList.add("mismatch");

  const content = document.createElement("span");
  const heading = document.createElement("strong");
  heading.textContent = sample.title || sample.sample_id;
  const comparison = document.createElement("small");
  comparison.textContent = `${sample.original_display_name} → ${sample.predicted_display_name || "예측 없음"}`;
  const status = document.createElement("small");
  status.textContent = sample.decision
    ? sample.decision.decision === "approved"
      ? "확정 완료"
      : "학습 제외"
    : `${reviewStatusText(sample.review_status)} · ${scoreText(sample.confidence)}`;
  content.append(heading, comparison, status);
  button.append(dot, content);
  return button;
}

function renderQueue() {
  ui.queueList.replaceChildren(...state.filtered.map(queueItem));
  if (!state.filtered.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "현재 필터에 남은 샘플이 없습니다.";
    empty.style.padding = "24px 16px";
    ui.queueList.append(empty);
  }
}

function selectedSample() {
  return state.payload.samples.find((sample) => sample.sample_id === state.selectedId) || null;
}

function renderTop3(sample) {
  const rows = sample.top3.map((candidate, index) => {
    const row = document.createElement("div");
    row.className = "top3-row";
    const rank = document.createElement("span");
    rank.className = "top3-rank";
    rank.textContent = String(index + 1);
    const body = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = `${candidate.display_name} · ${candidate.group_name}`;
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("span");
    const percent = Number.isFinite(candidate.score) ? Math.max(0, Math.min(100, candidate.score * 100)) : 0;
    fill.style.width = `${percent}%`;
    track.append(fill);
    body.append(label, track);
    const score = document.createElement("span");
    score.className = "top3-score";
    score.textContent = scoreText(candidate.score);
    row.append(rank, body, score);
    return row;
  });
  ui.top3List.replaceChildren(...rows);
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "예측 결과 없음";
    ui.top3List.append(empty);
  }
}

function setReviewControls(sample) {
  const decision = sample.decision;
  ui.labelSelect.value = decision?.selected_leaf_id || sample.original_leaf_id;
  ui.decisionNote.value = decision?.note || "";
  ui.clearDecision.disabled = !decision;
  const controls = [ui.confirmOriginal, ui.saveCorrection, ui.rejectSample, ui.clearDecision];
  for (const control of controls) control.disabled = state.busy;
  if (!decision) ui.clearDecision.disabled = true;
}

function clearCurrent() {
  ui.sampleImage.removeAttribute("src");
  ui.sampleImage.hidden = true;
  ui.imageEmpty.hidden = false;
  for (const element of [
    ui.sampleId,
    ui.sampleTitle,
    ui.sampleSource,
    ui.sampleQuery,
    ui.sampleLicense,
    ui.sampleSplit,
    ui.sampleModel,
    ui.originalLabel,
    ui.originalLeaf,
    ui.predictedLabel,
    ui.predictedLeaf,
  ]) {
    element.textContent = "-";
  }
  ui.sampleStatus.textContent = "-";
  ui.sampleStatus.className = "status-badge";
  ui.top3List.replaceChildren();
  ui.labelSelect.disabled = true;
  ui.decisionNote.disabled = true;
  for (const button of [ui.confirmOriginal, ui.saveCorrection, ui.rejectSample, ui.clearDecision]) {
    button.disabled = true;
  }
  ui.queuePosition.textContent = "0 / 0";
  ui.previousButton.disabled = true;
  ui.nextButton.disabled = true;
}

function renderCurrent() {
  const sample = selectedSample();
  if (!sample) {
    clearCurrent();
    return;
  }
  const index = state.filtered.findIndex((item) => item.sample_id === sample.sample_id);
  ui.queuePosition.textContent = `${index + 1} / ${state.filtered.length}`;
  ui.previousButton.disabled = state.busy || index <= 0;
  ui.nextButton.disabled = state.busy || index < 0 || index >= state.filtered.length - 1;
  ui.labelSelect.disabled = false;
  ui.decisionNote.disabled = false;

  if (sample.image_url) {
    ui.sampleImage.src = sample.image_url;
    ui.sampleImage.hidden = false;
    ui.imageEmpty.hidden = true;
  } else {
    ui.sampleImage.removeAttribute("src");
    ui.sampleImage.hidden = true;
    ui.imageEmpty.hidden = false;
  }
  ui.sampleId.textContent = sample.sample_id;
  ui.sampleTitle.textContent = text(sample.title);
  ui.sampleSource.textContent = text(sample.source);
  ui.sampleQuery.textContent = text(sample.query);
  ui.sampleLicense.textContent = text(sample.license);
  ui.sampleSplit.textContent = text(sample.dataset_split);
  ui.sampleModel.textContent = text(sample.model_version);
  ui.originalLabel.textContent = `${sample.original_group_name} · ${sample.original_display_name}`;
  ui.originalLeaf.textContent = sample.original_leaf_id;
  ui.predictedLabel.textContent = sample.predicted_display_name
    ? `${sample.predicted_group_name} · ${sample.predicted_display_name}`
    : "예측 없음";
  ui.predictedLeaf.textContent = text(sample.predicted_leaf_id);
  ui.sampleStatus.textContent = sample.mismatch ? "불일치" : "일치";
  ui.sampleStatus.className = `status-badge ${sample.mismatch ? "mismatch" : "match"}`;
  renderTop3(sample);
  setReviewControls(sample);
}

function replaceSample(updated) {
  const index = state.payload.samples.findIndex((sample) => sample.sample_id === updated.sample_id);
  if (index >= 0) state.payload.samples[index] = updated;
}

function currentRevision(sample) {
  return Number(sample?.decision?.revision || 0);
}

async function saveDecision(decision, selectedLeafId = null) {
  const sample = selectedSample();
  if (!sample || state.busy) return;
  state.busy = true;
  renderCurrent();
  ui.saveStatus.textContent = "저장 중";
  try {
    const result = await request(`/api/decisions/${encodeURIComponent(sample.sample_id)}`, {
      method: "PUT",
      body: JSON.stringify({
        decision,
        selected_leaf_id: selectedLeafId,
        note: ui.decisionNote.value,
        expected_revision: currentRevision(sample),
      }),
    });
    replaceSample(result.sample);
    state.payload.summary = result.summary;
    updateMetrics(result.summary);
    const previousIndex = state.filtered.findIndex((item) => item.sample_id === sample.sample_id);
    applyFilters({ preserveSelection: false });
    if (state.filtered.length && previousIndex >= 0) {
      const nextIndex = Math.min(previousIndex, state.filtered.length - 1);
      state.selectedId = state.filtered[nextIndex].sample_id;
      renderQueue();
      renderCurrent();
    }
    const message =
      decision === "approved"
        ? selectedLeafId === sample.original_leaf_id
          ? "원본 라벨을 확정했습니다."
          : "정답 라벨을 교정했습니다."
        : decision === "rejected"
          ? "학습 대상에서 제외했습니다."
          : "검수 결정을 취소했습니다.";
    showToast(message);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.busy = false;
    updateMetrics(state.payload.summary);
    renderCurrent();
  }
}

function move(delta) {
  const index = state.filtered.findIndex((sample) => sample.sample_id === state.selectedId);
  const target = state.filtered[index + delta];
  if (!target) return;
  state.selectedId = target.sample_id;
  renderQueue();
  renderCurrent();
  document.querySelector(`[data-sample-id="${CSS.escape(target.sample_id)}"]`)?.scrollIntoView({
    block: "nearest",
  });
}

async function exportTrainingData() {
  if (state.busy) return;
  state.busy = true;
  ui.exportButton.disabled = true;
  ui.saveStatus.textContent = "생성 중";
  try {
    const result = await request("/api/export", { method: "POST" });
    showToast(`재학습 manifest 생성 완료\n${result.training_manifest}`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.busy = false;
    ui.exportButton.disabled = false;
    updateMetrics(state.payload.summary);
  }
}

function bindEvents() {
  for (const tab of ui.tabs) {
    tab.addEventListener("click", () => {
      state.mode = tab.dataset.mode;
      for (const item of ui.tabs) {
        item.setAttribute("aria-selected", String(item === tab));
      }
      applyFilters({ preserveSelection: false });
    });
  }
  ui.leafFilter.addEventListener("change", () => applyFilters({ preserveSelection: false }));
  ui.searchInput.addEventListener("input", () => applyFilters({ preserveSelection: false }));
  ui.previousButton.addEventListener("click", () => move(-1));
  ui.nextButton.addEventListener("click", () => move(1));
  ui.confirmOriginal.addEventListener("click", () => {
    const sample = selectedSample();
    if (sample) saveDecision("approved", sample.original_leaf_id);
  });
  ui.saveCorrection.addEventListener("click", () => saveDecision("approved", ui.labelSelect.value));
  ui.rejectSample.addEventListener("click", () => saveDecision("rejected"));
  ui.clearDecision.addEventListener("click", () => saveDecision("cleared"));
  ui.exportButton.addEventListener("click", exportTrainingData);
  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) {
      return;
    }
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });
}

async function start() {
  bindEvents();
  try {
    state.payload = await request("/api/state");
    populateCategoryMenus();
    updateMetrics(state.payload.summary);
    applyFilters({ preserveSelection: false });
  } catch (error) {
    ui.saveStatus.textContent = "불러오기 실패";
    showToast(error.message, true);
    clearCurrent();
  }
}

start();
