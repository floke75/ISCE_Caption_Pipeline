import { useState } from "react";
import {
  useJobs,
  useLaunchInference,
  useLaunchTrainingPair,
  useLaunchModelTraining,
  JobResponse,
  TrainModelRequest,
} from "./api";
import { SectionCard } from "./components/SectionCard";
import { FormField } from "./components/FormField";
import { JobTable } from "./components/JobTable";
import { LogViewer } from "./components/LogViewer";
import { ConfigEditor } from "./components/ConfigEditor";

function Hero() {
  return (
    <header className="mb-10 flex flex-col space-y-6 rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 p-10 shadow-2xl shadow-slate-900/60">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold text-slate-50">ISCE Captioning Control Panel</h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-400">
            Launch manual inference, prep new training examples, and kick off model training runs without touching the command
            line. Stream live logs, tweak configuration files, and keep the production pipeline in sync.
          </p>
        </div>
      </div>
      <div className="grid gap-4 text-xs text-slate-400 md:grid-cols-3">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-sm font-semibold text-slate-200">Manual inference</div>
          <p>Upload TXT transcripts and media, then monitor segmentation output with live diagnostics.</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-sm font-semibold text-slate-200">Training data factory</div>
          <p>Align gold-standard SRTs with ASR output and generate enriched JSON assets for the trainer.</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="text-sm font-semibold text-slate-200">Iterative reweighting</div>
          <p>Run the statistical training loop with curated corpora and pull updated model weights.</p>
        </div>
      </div>
    </header>
  );
}

function InferenceForm() {
  const [mediaPath, setMediaPath] = useState("");
  const [transcriptPath, setTranscriptPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [pipelineRoot, setPipelineRoot] = useState("");
  const launchInference = useLaunchInference();

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const overrides: Record<string, string> = {};
    if (outputDir) overrides["output_dir"] = outputDir;
    if (pipelineRoot) overrides["pipeline_root"] = pipelineRoot;
    launchInference.mutate({
      media_path: mediaPath,
      transcript_path: transcriptPath || undefined,
      pipeline_overrides: Object.keys(overrides).length ? overrides : undefined,
    });
  };

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <FormField label="Media file" description="Path to .mp4/.wav/etc.">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={mediaPath}
          onChange={(event) => setMediaPath(event.target.value)}
          placeholder="/data/inference/MyClip.mp4"
          required
        />
      </FormField>
      <FormField label="Transcript" description="Optional TXT transcript for guided inference.">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={transcriptPath}
          onChange={(event) => setTranscriptPath(event.target.value)}
          placeholder="/data/inference/MyClip.txt"
        />
      </FormField>
      <div className="grid gap-4 sm:grid-cols-2">
        <FormField label="Output directory" description="Defaults to pipeline_config output_dir.">
          <input
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            value={outputDir}
            onChange={(event) => setOutputDir(event.target.value)}
            placeholder="/data/runtime/output"
          />
        </FormField>
        <FormField label="Pipeline root" description="Overrides runtime pipeline_root.">
          <input
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            value={pipelineRoot}
            onChange={(event) => setPipelineRoot(event.target.value)}
            placeholder="/data/runtime"
          />
        </FormField>
      </div>
      <button
        type="submit"
        className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-brand-500/30 transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={launchInference.isLoading}
      >
        {launchInference.isLoading ? "Launching…" : "Start inference"}
      </button>
      {launchInference.isError && (
        <div className="text-sm text-rose-300">{(launchInference.error as Error).message}</div>
      )}
      {launchInference.isSuccess && (
        <div className="text-sm text-emerald-300">Inference job queued successfully.</div>
      )}
    </form>
  );
}

function TrainingPairForm() {
  const [mediaPath, setMediaPath] = useState("");
  const [srtPath, setSrtPath] = useState("");
  const [pipelineRoot, setPipelineRoot] = useState("");
  const launchTrainingPair = useLaunchTrainingPair();

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const overrides: Record<string, string> = {};
    if (pipelineRoot) overrides["pipeline_root"] = pipelineRoot;
    launchTrainingPair.mutate({
      media_path: mediaPath,
      srt_path: srtPath,
      pipeline_overrides: Object.keys(overrides).length ? overrides : undefined,
    });
  };

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <FormField label="Media file">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={mediaPath}
          onChange={(event) => setMediaPath(event.target.value)}
          placeholder="/data/training/MyClip.mp4"
          required
        />
      </FormField>
      <FormField label="Reference SRT">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={srtPath}
          onChange={(event) => setSrtPath(event.target.value)}
          placeholder="/data/training/MyClip.srt"
          required
        />
      </FormField>
      <FormField label="Pipeline root" description="Overrides runtime pipeline root">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={pipelineRoot}
          onChange={(event) => setPipelineRoot(event.target.value)}
        />
      </FormField>
      <button
        type="submit"
        className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-brand-500/30 transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={launchTrainingPair.isLoading}
      >
        {launchTrainingPair.isLoading ? "Launching…" : "Create training pair"}
      </button>
      {launchTrainingPair.isError && (
        <div className="text-sm text-rose-300">{(launchTrainingPair.error as Error).message}</div>
      )}
      {launchTrainingPair.isSuccess && (
        <div className="text-sm text-emerald-300">Training pair job queued.</div>
      )}
    </form>
  );
}

function TrainingLoopForm() {
  const [form, setForm] = useState<TrainModelRequest>({
    corpus_dir: "",
    constraints_path: "",
    weights_path: "",
    iterations: 3,
    error_boost_factor: 1,
    config_path: "config.yaml",
  });
  const launchModelTraining = useLaunchModelTraining();

  const handleChange = (key: keyof TrainModelRequest, value: string) => {
    setForm((prev) => ({
      ...prev,
      [key]: key === "iterations" || key === "error_boost_factor" ? Number(value) : value,
    }));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    launchModelTraining.mutate(form);
  };

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <FormField label="Corpus directory" description="Directory with *.json training files">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={form.corpus_dir}
          onChange={(event) => handleChange("corpus_dir", event.target.value)}
          required
        />
      </FormField>
      <div className="grid gap-4 sm:grid-cols-2">
        <FormField label="constraints.json output">
          <input
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            value={form.constraints_path}
            onChange={(event) => handleChange("constraints_path", event.target.value)}
            required
          />
        </FormField>
        <FormField label="model_weights.json output">
          <input
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            value={form.weights_path}
            onChange={(event) => handleChange("weights_path", event.target.value)}
            required
          />
        </FormField>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <FormField label="Iterations">
          <input
            type="number"
            min={1}
            max={20}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            value={form.iterations}
            onChange={(event) => handleChange("iterations", event.target.value)}
          />
        </FormField>
        <FormField label="Error boost factor">
          <input
            type="number"
            step="0.1"
            min={0}
            max={10}
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            value={form.error_boost_factor}
            onChange={(event) => handleChange("error_boost_factor", event.target.value)}
          />
        </FormField>
      </div>
      <FormField label="Config path">
        <input
          className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
          value={form.config_path}
          onChange={(event) => handleChange("config_path", event.target.value)}
        />
      </FormField>
      <button
        type="submit"
        className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-brand-500/30 transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={launchModelTraining.isLoading}
      >
        {launchModelTraining.isLoading ? "Launching…" : "Start training loop"}
      </button>
      {launchModelTraining.isError && (
        <div className="text-sm text-rose-300">{(launchModelTraining.error as Error).message}</div>
      )}
      {launchModelTraining.isSuccess && (
        <div className="text-sm text-emerald-300">Training loop queued.</div>
      )}
    </form>
  );
}

export default function App() {
  const { data: jobsData } = useJobs();
  const [selectedJob, setSelectedJob] = useState<JobResponse | null>(null);

  return (
    <div className="min-h-screen bg-slate-950 pb-16">
      <div className="mx-auto w-full max-w-6xl px-6 py-10">
        <Hero />
        <div className="grid gap-6 lg:grid-cols-2">
          <SectionCard
            title="Manual inference"
            subtitle="Run the end-to-end captioning stack on demand"
            className="lg:row-span-1"
          >
            <InferenceForm />
          </SectionCard>
          <SectionCard title="Generate training pairs" subtitle="Build enriched JSON corpora">
            <TrainingPairForm />
          </SectionCard>
          <SectionCard title="Run iterative model training" subtitle="Launch scripts/train_model.py">
            <TrainingLoopForm />
          </SectionCard>
          <SectionCard title="Pipeline configuration" subtitle="Edit YAML overrides without a text editor">
            <ConfigEditor configKey="pipeline" />
          </SectionCard>
          <SectionCard title="Caption engine configuration" subtitle="config.yaml">
            <ConfigEditor configKey="core" />
          </SectionCard>
        </div>
        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          <SectionCard title="Job history" subtitle="Click a row to inspect log output">
            <JobTable
              jobs={jobsData?.jobs ?? []}
              selectedJobId={selectedJob?.id}
              onSelect={(job) => setSelectedJob(job)}
            />
          </SectionCard>
          <SectionCard title="Live logs" subtitle="Automatically refreshes every two seconds">
            <LogViewer jobId={selectedJob?.id ?? null} />
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
