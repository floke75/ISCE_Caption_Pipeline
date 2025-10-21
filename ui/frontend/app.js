const API_BASE = "/api";
const pollingIntervalMs = 5000;

const inferenceForm = document.querySelector("#inference-form");
const trainingPairForm = document.querySelector("#training-pair-form");
const modelTrainingForm = document.querySelector("#model-training-form");
const pipelineConfigForm = document.querySelector("#pipeline-config-form");
const modelConfigForm = document.querySelector("#model-config-form");
const refreshConfigButton = document.querySelector("#refresh-config");
const refreshJobsButton = document.querySelector("#refresh-jobs");
const jobsTableBody = document.querySelector("#jobs-table tbody");
const jobLogs = document.querySelector("#job-logs");
const jobMeta = document.querySelector("#job-meta");
const configStatus = document.querySelector("#config-status");

let selectedJobId = null;
let pollingHandle = null;

function showToast(message, tone = "info") {
  const template = document.querySelector("#toast-template");
  if (!template) return;
  const toast = template.content.firstElementChild.cloneNode(true);
  toast.textContent = message;
  toast.dataset.tone = tone;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

async function fetchJSON(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || response.statusText);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function formDataToObject(form) {
  const data = new FormData(form);
  const payload = {};
  data.forEach((value, key) => {
    if (value === "" || value == null) {
      return;
    }
    payload[key] = value;
  });
  return payload;
}

function valueOrNull(value) {
  if (value === "" || value == null) return null;
  return value;
}

function boolFromCheckbox(input) {
  return input?.checked ?? false;
}

async function handleSubmit(form, endpoint) {
  const payload = formDataToObject(form);
  try {
    const job = await fetchJSON(endpoint, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    showToast(`Job ${job.id} queued`);
    form.reset();
    await refreshJobs();
  } catch (error) {
    console.error(error);
    showToast(`Failed to start job: ${error.message}`, "error");
  }
}

inferenceForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  handleSubmit(inferenceForm, "/jobs/inference");
});

trainingPairForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  handleSubmit(trainingPairForm, "/jobs/training-pair");
});

modelTrainingForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  handleSubmit(modelTrainingForm, "/jobs/model-training");
});

refreshConfigButton?.addEventListener("click", async () => {
  await loadConfiguration();
  showToast("Configuration refreshed");
});

pipelineConfigForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = buildPipelineConfigPayload();
  try {
    await fetchJSON("/config/pipeline", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    configStatus.textContent = "Pipeline configuration saved.";
    showToast("Pipeline configuration saved", "success");
  } catch (error) {
    configStatus.textContent = "Failed to save pipeline configuration.";
    showToast(error.message, "error");
  }
});

modelConfigForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = buildModelConfigPayload();
  try {
    await fetchJSON("/config/model", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    configStatus.textContent = "Model configuration saved.";
    showToast("Model configuration saved", "success");
  } catch (error) {
    configStatus.textContent = "Failed to save model configuration.";
    showToast(error.message, "error");
  }
});

refreshJobsButton?.addEventListener("click", async () => {
  await refreshJobs();
  if (selectedJobId) await loadJobDetail(selectedJobId);
});

async function loadConfiguration() {
  try {
    const [pipelineConfig, modelConfig] = await Promise.all([
      fetchJSON("/config/pipeline"),
      fetchJSON("/config/model"),
    ]);
    populatePipelineConfigForm(pipelineConfig);
    populateModelConfigForm(modelConfig);
    configStatus.textContent = "";
  } catch (error) {
    configStatus.textContent = "Unable to load configuration.";
    console.error(error);
  }
}

function populatePipelineConfigForm(data) {
  const map = (name) => pipelineConfigForm?.querySelector(`[name="${name}"]`);
  map("project_root").value = data.project_root ?? "";
  map("pipeline_root").value = data.pipeline_root ?? "";
  map("drop_folder_inference").value = data.drop_folder_inference ?? "";
  map("drop_folder_training").value = data.drop_folder_training ?? "";
  map("srt_placement_folder").value = data.srt_placement_folder ?? "";
  map("txt_placement_folder").value = data.txt_placement_folder ?? "";
  map("intermediate_dir").value = data.intermediate_dir ?? "";
  map("output_dir").value = data.output_dir ?? "";
  map("processed_dir").value = data.processed_dir ?? "";

  const align = data.align_make ?? {};
  map("whisper_model_id").value = align.whisper_model_id ?? "";
  map("align_model_id").value = align.align_model_id ?? "";
  map("align_language").value = align.language ?? "";
  map("compute_type").value = align.compute_type ?? "";
  map("batch_size").value = align.batch_size ?? "";
  map("hf_token").value = align.hf_token ?? "";
  map("do_diarization").checked = !!align.do_diarization;
  map("skip_if_asr_exists").checked = !!align.skip_if_asr_exists;

  const build = data.build_pair ?? {};
  map("build_language").value = build.language ?? "";
  map("time_tolerance_s").value = build.time_tolerance_s ?? "";
  map("round_seconds").value = build.round_seconds ?? "";
  map("spacy_enable").checked = !!build.spacy_enable;
  map("spacy_model").value = build.spacy_model ?? "";
  map("spacy_add_dependencies").checked = !!build.spacy_add_dependencies;
  map("emit_asr_style_training_copy").checked = !!build.emit_asr_style_training_copy;
  map("txt_match_close").value = build.txt_match_close ?? "";
  map("txt_match_weak").value = build.txt_match_weak ?? "";

  const orch = data.orchestrator ?? {};
  map("poll_interval_seconds").value = orch.poll_interval_seconds ?? "";
  map("file_settle_delay_seconds").value = orch.file_settle_delay_seconds ?? "";
  map("srt_wait_timeout_seconds").value = orch.srt_wait_timeout_seconds ?? "";
}

function populateModelConfigForm(data) {
  const map = (name) => modelConfigForm?.querySelector(`[name="${name}"]`);
  map("beam_width").value = data.beam_width ?? "";
  const constraints = data.constraints ?? {};
  map("min_block_duration_s").value = constraints.min_block_duration_s ?? "";
  map("max_block_duration_s").value = constraints.max_block_duration_s ?? "";
  map("line_length_soft_target").value = constraints.line_length_soft_target ?? "";
  map("line_length_hard_limit").value = constraints.line_length_hard_limit ?? "";
  map("min_chars_for_single_word_block").value = constraints.min_chars_for_single_word_block ?? "";

  const sliders = data.sliders ?? {};
  map("flow").value = sliders.flow ?? "";
  map("density").value = sliders.density ?? "";
  map("balance").value = sliders.balance ?? "";
  map("line_length_leniency").value = sliders.line_length_leniency ?? "";
  map("orphan_leniency").value = sliders.orphan_leniency ?? "";
  map("structure_boost").value = sliders.structure_boost ?? "";

  const paths = data.paths ?? {};
  map("model_weights").value = paths.model_weights ?? "";
  map("constraints").value = paths.constraints ?? "";
}

function buildPipelineConfigPayload() {
  const payload = {
    project_root: valueOrNull(pipelineConfigForm.project_root.value),
    pipeline_root: valueOrNull(pipelineConfigForm.pipeline_root.value),
    drop_folder_inference: valueOrNull(pipelineConfigForm.drop_folder_inference.value),
    drop_folder_training: valueOrNull(pipelineConfigForm.drop_folder_training.value),
    srt_placement_folder: valueOrNull(pipelineConfigForm.srt_placement_folder.value),
    txt_placement_folder: valueOrNull(pipelineConfigForm.txt_placement_folder.value),
    intermediate_dir: valueOrNull(pipelineConfigForm.intermediate_dir.value),
    output_dir: valueOrNull(pipelineConfigForm.output_dir.value),
    processed_dir: valueOrNull(pipelineConfigForm.processed_dir.value),
    align_make: {
      whisper_model_id: valueOrNull(pipelineConfigForm.whisper_model_id.value),
      align_model_id: valueOrNull(pipelineConfigForm.align_model_id.value),
      language: valueOrNull(pipelineConfigForm.align_language.value),
      compute_type: valueOrNull(pipelineConfigForm.compute_type.value),
      batch_size: pipelineConfigForm.batch_size.value ? Number(pipelineConfigForm.batch_size.value) : null,
      hf_token: valueOrNull(pipelineConfigForm.hf_token.value),
      do_diarization: boolFromCheckbox(pipelineConfigForm.do_diarization),
      skip_if_asr_exists: boolFromCheckbox(pipelineConfigForm.skip_if_asr_exists),
    },
    build_pair: {
      language: valueOrNull(pipelineConfigForm.build_language.value),
      time_tolerance_s: pipelineConfigForm.time_tolerance_s.value ? Number(pipelineConfigForm.time_tolerance_s.value) : null,
      round_seconds: pipelineConfigForm.round_seconds.value ? Number(pipelineConfigForm.round_seconds.value) : null,
      spacy_enable: boolFromCheckbox(pipelineConfigForm.spacy_enable),
      spacy_model: valueOrNull(pipelineConfigForm.spacy_model.value),
      spacy_add_dependencies: boolFromCheckbox(pipelineConfigForm.spacy_add_dependencies),
      emit_asr_style_training_copy: boolFromCheckbox(pipelineConfigForm.emit_asr_style_training_copy),
      txt_match_close: pipelineConfigForm.txt_match_close.value ? Number(pipelineConfigForm.txt_match_close.value) : null,
      txt_match_weak: pipelineConfigForm.txt_match_weak.value ? Number(pipelineConfigForm.txt_match_weak.value) : null,
    },
    orchestrator: {
      poll_interval_seconds: pipelineConfigForm.poll_interval_seconds.value ? Number(pipelineConfigForm.poll_interval_seconds.value) : null,
      file_settle_delay_seconds: pipelineConfigForm.file_settle_delay_seconds.value ? Number(pipelineConfigForm.file_settle_delay_seconds.value) : null,
      srt_wait_timeout_seconds: pipelineConfigForm.srt_wait_timeout_seconds.value ? Number(pipelineConfigForm.srt_wait_timeout_seconds.value) : null,
    },
  };

  Object.keys(payload.align_make).forEach((key) => payload.align_make[key] == null && delete payload.align_make[key]);
  Object.keys(payload.build_pair).forEach((key) => payload.build_pair[key] == null && delete payload.build_pair[key]);
  Object.keys(payload.orchestrator).forEach((key) => payload.orchestrator[key] == null && delete payload.orchestrator[key]);
  Object.keys(payload).forEach((key) => {
    if (payload[key] == null || (typeof payload[key] === "object" && Object.keys(payload[key]).length === 0)) {
      delete payload[key];
    }
  });
  return payload;
}

function buildModelConfigPayload() {
  const payload = {
    beam_width: modelConfigForm.beam_width.value ? Number(modelConfigForm.beam_width.value) : null,
    constraints: {
      min_block_duration_s: modelConfigForm.min_block_duration_s.value ? Number(modelConfigForm.min_block_duration_s.value) : null,
      max_block_duration_s: modelConfigForm.max_block_duration_s.value ? Number(modelConfigForm.max_block_duration_s.value) : null,
      line_length_soft_target: modelConfigForm.line_length_soft_target.value ? Number(modelConfigForm.line_length_soft_target.value) : null,
      line_length_hard_limit: modelConfigForm.line_length_hard_limit.value ? Number(modelConfigForm.line_length_hard_limit.value) : null,
      min_chars_for_single_word_block: modelConfigForm.min_chars_for_single_word_block.value ? Number(modelConfigForm.min_chars_for_single_word_block.value) : null,
    },
    sliders: {
      flow: modelConfigForm.flow.value ? Number(modelConfigForm.flow.value) : null,
      density: modelConfigForm.density.value ? Number(modelConfigForm.density.value) : null,
      balance: modelConfigForm.balance.value ? Number(modelConfigForm.balance.value) : null,
      line_length_leniency: modelConfigForm.line_length_leniency.value ? Number(modelConfigForm.line_length_leniency.value) : null,
      orphan_leniency: modelConfigForm.orphan_leniency.value ? Number(modelConfigForm.orphan_leniency.value) : null,
      structure_boost: modelConfigForm.structure_boost.value ? Number(modelConfigForm.structure_boost.value) : null,
    },
    paths: {
      model_weights: valueOrNull(modelConfigForm.model_weights.value),
      constraints: valueOrNull(modelConfigForm.constraints.value),
    },
  };
  Object.keys(payload.constraints).forEach((key) => payload.constraints[key] == null && delete payload.constraints[key]);
  Object.keys(payload.sliders).forEach((key) => payload.sliders[key] == null && delete payload.sliders[key]);
  Object.keys(payload.paths).forEach((key) => payload.paths[key] == null && delete payload.paths[key]);
  Object.keys(payload).forEach((key) => {
    if (payload[key] == null || (typeof payload[key] === "object" && Object.keys(payload[key]).length === 0)) {
      delete payload[key];
    }
  });
  return payload;
}

async function refreshJobs() {
  try {
    const jobs = await fetchJSON("/jobs");
    renderJobTable(jobs);
  } catch (error) {
    console.error(error);
  }
}

function renderJobTable(jobs) {
  jobsTableBody.innerHTML = "";
  jobs
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
    .forEach((job) => {
      const row = document.createElement("tr");
      row.dataset.jobId = job.id;
      if (job.id === selectedJobId) {
        row.classList.add("active");
      }
      row.innerHTML = `
        <td>${job.id.slice(0, 8)}</td>
        <td>${job.type.replace("_", " ")}</td>
        <td><span class="status-pill" data-status="${job.status}">${job.status}</span></td>
        <td>${renderProgress(job.progress)}</td>
        <td>${formatRelativeTime(job.updated_at)}</td>
      `;
      row.addEventListener("click", () => selectJob(job.id));
      jobsTableBody.appendChild(row);
    });
}

function renderProgress(value) {
  if (value == null) return "–";
  const percentage = Math.round((value || 0) * 100);
  return `<progress max="100" value="${percentage}"></progress>`;
}

function formatRelativeTime(dateString) {
  const date = new Date(dateString);
  const diff = Date.now() - date.getTime();
  const minutes = Math.round(diff / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
}

async function selectJob(jobId) {
  selectedJobId = jobId;
  await loadJobDetail(jobId);
  await refreshJobs();
}

async function loadJobDetail(jobId) {
  try {
    const [job, logPayload] = await Promise.all([
      fetchJSON(`/jobs/${jobId}`),
      fetchJSON(`/jobs/${jobId}/logs`),
    ]);
    renderJobMeta(job);
    jobLogs.textContent = logPayload.log || "(no logs yet)";
    jobLogs.scrollTop = jobLogs.scrollHeight;
  } catch (error) {
    console.error(error);
    jobMeta.textContent = "Unable to load job.";
    jobLogs.textContent = "";
  }
}

function renderJobMeta(job) {
  const entries = [
    ["Job ID", job.id],
    ["Type", job.type],
    ["Status", job.status],
    ["Created", new Date(job.created_at).toLocaleString()],
    ["Updated", new Date(job.updated_at).toLocaleString()],
  ];
  if (job.progress != null) {
    entries.push(["Progress", `${Math.round(job.progress * 100)}%`]);
  }
  if (job.metrics) {
    Object.entries(job.metrics).forEach(([key, value]) => {
      entries.push([key.replace(/_/g, " "), value]);
    });
  }
  if (job.result && Object.keys(job.result).length) {
    Object.entries(job.result).forEach(([key, value]) => {
      entries.push([`Result: ${key.replace(/_/g, " ")}`, value]);
    });
  }
  while (jobMeta.firstChild) {
    jobMeta.removeChild(jobMeta.firstChild);
  }

  entries.forEach(([label, value]) => {
    const entry = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = `${label}:`;
    entry.appendChild(strong);

    if (value !== undefined && value !== null) {
      entry.append(" ");
      entry.append(document.createTextNode(String(value)));
    }

    jobMeta.appendChild(entry);
  });
}

function startPolling() {
  if (pollingHandle) clearInterval(pollingHandle);
  pollingHandle = setInterval(async () => {
    await refreshJobs();
    if (selectedJobId) await loadJobDetail(selectedJobId);
  }, pollingIntervalMs);
}

loadConfiguration();
refreshJobs();
startPolling();
