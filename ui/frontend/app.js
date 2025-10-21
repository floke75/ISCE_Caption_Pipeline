const inferenceForm = document.getElementById("inference-form");
const trainingPairForm = document.getElementById("training-pair-form");
const modelTrainingForm = document.getElementById("model-training-form");
const inferenceFeedback = document.getElementById("inference-feedback");
const trainingFeedback = document.getElementById("training-pair-feedback");
const modelTrainingFeedback = document.getElementById("model-training-feedback");
const configFeedback = document.getElementById("config-feedback");
const jobsTableBody = document.getElementById("jobs-table-body");
const jobLogViewer = document.getElementById("job-log-viewer");
const jobResult = document.getElementById("job-result");
const jobTitle = document.getElementById("selected-job-title");
const healthIndicator = document.getElementById("health-indicator");
const refreshJobsBtn = document.getElementById("refresh-jobs");
const pipelineEditor = document.getElementById("pipeline-config-editor");
const modelEditor = document.getElementById("model-config-editor");

let selectedJobId = null;

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  if (response.status === 204) {
    return {};
  }
  return response.json();
}

function setFeedback(element, message, variant = "") {
  element.textContent = message;
  element.classList.remove("success", "error");
  if (variant) {
    element.classList.add(variant);
  }
}

function parseJSONField(value) {
  if (!value || !value.trim()) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`Invalid JSON: ${error.message}`);
  }
}

async function submitInference(event) {
  event.preventDefault();
  try {
    const data = new FormData(inferenceForm);
    const payload = {
      media_path: data.get("media_path").trim(),
      transcript_path: data.get("transcript_path").trim() || null,
      output_path: data.get("output_path").trim() || null,
      config_overrides: parseJSONField(data.get("config_overrides")),
      pipeline_config_path: data.get("pipeline_config_path").trim() || null,
    };

    const response = await fetchJSON("/jobs/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setFeedback(inferenceFeedback, `Job queued: ${response.job_id}`, "success");
    inferenceForm.reset();
    await updateJobs();
  } catch (error) {
    console.error(error);
    setFeedback(inferenceFeedback, error.message, "error");
  }
}

async function submitTrainingPair(event) {
  event.preventDefault();
  try {
    const data = new FormData(trainingPairForm);
    const payload = {
      media_path: data.get("media_path").trim(),
      srt_path: data.get("srt_path").trim(),
      config_overrides: parseJSONField(data.get("config_overrides")),
      pipeline_config_path: data.get("pipeline_config_path").trim() || null,
    };

    const response = await fetchJSON("/jobs/training-pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setFeedback(trainingFeedback, `Job queued: ${response.job_id}`, "success");
    trainingPairForm.reset();
    await updateJobs();
  } catch (error) {
    console.error(error);
    setFeedback(trainingFeedback, error.message, "error");
  }
}

async function submitModelTraining(event) {
  event.preventDefault();
  try {
    const data = new FormData(modelTrainingForm);
    const payload = {
      corpus_dir: data.get("corpus_dir").trim(),
      iterations: Number(data.get("iterations")) || 3,
      error_boost_factor: Number(data.get("error_boost_factor")) || 1.0,
      constraints_output: data.get("constraints_output").trim() || null,
      weights_output: data.get("weights_output").trim() || null,
      config_path: data.get("config_path").trim() || null,
    };

    const response = await fetchJSON("/jobs/model-training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setFeedback(modelTrainingFeedback, `Job queued: ${response.job_id}`, "success");
    modelTrainingForm.reset();
    modelTrainingForm.elements["iterations"].value = 3;
    modelTrainingForm.elements["error_boost_factor"].value = 1.0;
    await updateJobs();
  } catch (error) {
    console.error(error);
    setFeedback(modelTrainingFeedback, error.message, "error");
  }
}

async function updateJobs() {
  try {
    const data = await fetchJSON("/jobs");
    renderJobs(data.jobs);
    if (selectedJobId) {
      await loadJob(selectedJobId);
    }
  } catch (error) {
    console.error(error);
  }
}

function renderJobs(jobs) {
  jobsTableBody.innerHTML = "";
  for (const job of jobs) {
    const row = document.createElement("tr");
    row.dataset.jobId = job.id;
    if (job.id === selectedJobId) {
      row.classList.add("active");
    }
    row.innerHTML = `
      <td>${job.id}</td>
      <td>${job.kind}</td>
      <td>${job.status}</td>
      <td>${new Date(job.updated_at).toLocaleString()}</td>
    `;
    row.addEventListener("click", () => {
      selectedJobId = job.id;
      updateJobs();
    });
    jobsTableBody.appendChild(row);
  }
}

async function loadJob(jobId) {
  try {
    const [details, logs] = await Promise.all([
      fetchJSON(`/jobs/${jobId}`),
      fetchJSON(`/jobs/${jobId}/logs`),
    ]);

    jobTitle.textContent = `Job ${details.id} — ${details.kind} (${details.status})`;
    jobLogViewer.textContent = logs.logs || "";

    if (details.result && Object.keys(details.result).length) {
      const entries = Object.entries(details.result)
        .map(([key, value]) => `<div><strong>${key}:</strong> ${value}</div>`)
        .join("");
      jobResult.innerHTML = entries;
    } else {
      jobResult.textContent = "No artifacts recorded yet.";
    }
  } catch (error) {
    console.error(error);
    jobLogViewer.textContent = error.message;
  }
}

async function refreshHealth() {
  try {
    await fetchJSON("/health");
    healthIndicator.textContent = "ok";
    healthIndicator.classList.add("ok");
    healthIndicator.classList.remove("error");
  } catch (error) {
    healthIndicator.textContent = "offline";
    healthIndicator.classList.add("error");
    healthIndicator.classList.remove("ok");
  }
}

async function loadConfig(target) {
  try {
    const endpoint = target === "pipeline" ? "/config/pipeline" : "/config/model";
    const data = await fetchJSON(endpoint);
    if (target === "pipeline") {
      pipelineEditor.value = data.content || "";
    } else {
      modelEditor.value = data.content || "";
    }
    setFeedback(configFeedback, `${target} configuration loaded`, "success");
  } catch (error) {
    console.error(error);
    setFeedback(configFeedback, error.message, "error");
  }
}

async function saveConfig(target) {
  try {
    const endpoint = target === "pipeline" ? "/config/pipeline" : "/config/model";
    const content = target === "pipeline" ? pipelineEditor.value : modelEditor.value;
    await fetchJSON(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    setFeedback(configFeedback, `${target} configuration saved`, "success");
  } catch (error) {
    console.error(error);
    setFeedback(configFeedback, error.message, "error");
  }
}

function wireConfigButtons() {
  const buttons = document.querySelectorAll("#config-panel button[data-action]");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.target;
      const action = button.dataset.action;
      if (action === "load") {
        void loadConfig(target);
      } else {
        void saveConfig(target);
      }
    });
  });
}

inferenceForm.addEventListener("submit", submitInference);
trainingPairForm.addEventListener("submit", submitTrainingPair);
modelTrainingForm.addEventListener("submit", submitModelTraining);
refreshJobsBtn.addEventListener("click", () => void updateJobs());
wireConfigButtons();

void refreshHealth();
void updateJobs();
loadConfig("pipeline").catch(() => {});
loadConfig("model").catch(() => {});
setInterval(refreshHealth, 15000);
setInterval(updateJobs, 5000);
