from __future__ import annotations

import re
import sys
from pathlib import Path

from .config_store import ConfigStore
from .job_manager import JobManager, JobRecord
from .schemas import InferenceRequest, ModelTrainingRequest, TrainingPairsRequest


class PipelineTasks:
    def __init__(
        self,
        job_manager: JobManager,
        pipeline_store: ConfigStore,
        model_store: ConfigStore,
    ) -> None:
        self._jobs = job_manager
        self._pipeline_store = pipeline_store
        self._model_store = model_store
        self._repo_root = Path(__file__).resolve().parent.parent
        self._python = sys.executable

    def launch_inference(self, payload: InferenceRequest) -> JobRecord:
        params = {
            "mediaPath": str(payload.media_path),
            "transcriptPath": str(payload.transcript_path),
            "outputPath": str(payload.output_path) if payload.output_path else None,
            "outputBasename": payload.output_basename,
        }
        return self._jobs.start_job("inference", params, lambda ctx: self._run_inference(ctx, payload))

    def launch_training_pairs(self, payload: TrainingPairsRequest) -> JobRecord:
        params = {
            "transcriptPath": str(payload.transcript_path),
            "asrReferencePath": str(payload.asr_reference_path),
            "outputBasename": payload.output_basename,
            "asrOnlyMode": payload.asr_only_mode,
        }
        return self._jobs.start_job("training_pairs", params, lambda ctx: self._run_training_pairs(ctx, payload))

    def launch_model_training(self, payload: ModelTrainingRequest) -> JobRecord:
        params = {
            "corpusDir": str(payload.corpus_dir),
            "constraintsOutput": str(payload.constraints_output) if payload.constraints_output else None,
            "weightsOutput": str(payload.weights_output) if payload.weights_output else None,
            "iterations": payload.iterations,
            "errorBoostFactor": payload.error_boost_factor,
        }
        return self._jobs.start_job("model_training", params, lambda ctx: self._run_model_training(ctx, payload))

    def _run_inference(self, ctx, payload: InferenceRequest) -> None:
        if not payload.media_path.exists():
            raise FileNotFoundError(f"Media file not found: {payload.media_path}")
        if not payload.transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {payload.transcript_path}")

        ctx.log("Preparing runtime configuration for inference job...")
        pipeline_config, pipeline_config_path = self._pipeline_store.prepare_runtime_config(
            payload.pipeline_overrides,
            ctx.workspace,
            isolation=True,
            repo_root=self._repo_root,
        )
        _, model_config_path = self._model_store.prepare_runtime_config(
            payload.model_overrides,
            ctx.workspace,
            filename="config.yaml",
        )

        media_base = payload.output_basename or payload.media_path.stem
        align_out_root = Path(pipeline_config.get("align_make", {}).get("out_root", ctx.workspace / "pipeline" / "_intermediate"))
        inference_dir = Path(pipeline_config.get("build_pair", {}).get("out_inference_dir", ctx.workspace / "pipeline" / "_intermediate" / "_inference_input"))

        align_out_root.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)

        ctx.run_command(
            [
                self._python,
                str(self._repo_root / "align_make.py"),
                "--input-file",
                payload.media_path,
                "--out-root",
                align_out_root,
                "--config-file",
                pipeline_config_path,
            ],
            cwd=self._repo_root,
            stage="Generating ASR reference",
            progress_range=(0.0, 0.32),
        )

        asr_path = align_out_root / "_align" / f"{payload.media_path.stem}.asr.visual.words.diar.json"
        if not asr_path.exists():
            raise FileNotFoundError(f"Expected ASR output was not produced: {asr_path}")
        ctx.add_artifact("ASR reference", asr_path)

        build_args = [
            self._python,
            str(self._repo_root / "build_training_pair_standalone.py"),
            "--primary-input",
            payload.transcript_path,
            "--asr-reference",
            asr_path,
            "--out-inference-dir",
            inference_dir,
            "--config-file",
            pipeline_config_path,
            "--output-basename",
            media_base,
        ]
        ctx.run_command(
            build_args,
            cwd=self._repo_root,
            stage="Aligning transcript and engineering features",
            progress_range=(0.32, 0.66),
        )

        enriched_path = inference_dir / f"{media_base}.enriched.json"
        if not enriched_path.exists():
            raise FileNotFoundError(f"Expected enriched output missing: {enriched_path}")
        ctx.add_artifact("Enriched tokens", enriched_path)

        artifacts_dir = ctx.workspace / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(payload.output_path) if payload.output_path else artifacts_dir / f"{media_base}.srt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        segmentation_args = [
            self._python,
            str(self._repo_root / "main.py"),
            "--input",
            enriched_path,
            "--output",
            output_path,
            "--config",
            model_config_path,
        ]
        ctx.run_command(
            segmentation_args,
            cwd=self._repo_root,
            stage="Generating final subtitles",
            progress_range=(0.66, 1.0),
        )

        if not output_path.exists():
            raise FileNotFoundError(f"Segmentation did not produce an SRT file: {output_path}")
        ctx.add_artifact("Segmented subtitles", output_path)
        ctx.set_result(output=str(output_path))

    def _run_training_pairs(self, ctx, payload: TrainingPairsRequest) -> None:
        if not payload.transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {payload.transcript_path}")
        if not payload.asr_reference_path.exists():
            raise FileNotFoundError(f"ASR reference file not found: {payload.asr_reference_path}")

        ctx.log("Preparing runtime configuration for training pair generation...")
        pipeline_config, pipeline_config_path = self._pipeline_store.prepare_runtime_config(
            payload.pipeline_overrides,
            ctx.workspace,
            isolation=True,
            repo_root=self._repo_root,
        )

        training_dir = Path(pipeline_config.get("build_pair", {}).get("out_training_dir", ctx.workspace / "pipeline" / "_intermediate" / "_training"))
        training_dir.mkdir(parents=True, exist_ok=True)

        base_name = payload.output_basename or payload.transcript_path.stem
        args = [
            self._python,
            str(self._repo_root / "build_training_pair_standalone.py"),
            "--primary-input",
            payload.transcript_path,
            "--asr-reference",
            payload.asr_reference_path,
            "--out-training-dir",
            training_dir,
            "--config-file",
            pipeline_config_path,
            "--output-basename",
            base_name,
        ]
        if payload.asr_only_mode:
            args.append("--asr-only-mode")
        ctx.run_command(
            args,
            cwd=self._repo_root,
            stage="Building training datasets",
            progress_range=(0.0, 1.0),
        )

        edited = training_dir / f"{base_name}.train.words.json"
        simulated = training_dir / f"{base_name}.train.raw.words.json"
        if edited.exists():
            ctx.add_artifact("Edited training JSON", edited)
        if simulated.exists():
            ctx.add_artifact("Simulated ASR training JSON", simulated)
        ctx.set_result(training_dir=str(training_dir))

    def _run_model_training(self, ctx, payload: ModelTrainingRequest) -> None:
        if not payload.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {payload.corpus_dir}")

        ctx.log("Preparing configuration for model training...")
        _, model_config_path = self._model_store.prepare_runtime_config(
            payload.model_overrides,
            ctx.workspace,
            filename="config.yaml",
        )

        artifacts_dir = ctx.workspace / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        constraints_path = Path(payload.constraints_output) if payload.constraints_output else artifacts_dir / "constraints.json"
        weights_path = Path(payload.weights_output) if payload.weights_output else artifacts_dir / "model_weights.json"
        constraints_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        pattern = re.compile(r"--- Starting Training Iteration (\d+)/(\d+) ---")

        def handle_output(line: str) -> None:
            match = pattern.search(line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2)) if match.group(2) else payload.iterations
                progress = 0.05 + (current - 1) / max(total, 1) * 0.9
                ctx.set_stage(f"Training iteration {current}/{total}")
                ctx.set_progress(min(progress, 0.95))

        args = [
            self._python,
            str(self._repo_root / "scripts" / "train_model.py"),
            "--corpus",
            payload.corpus_dir,
            "--constraints",
            constraints_path,
            "--weights",
            weights_path,
            "--config",
            model_config_path,
            "--iterations",
            str(payload.iterations),
            "--error-boost-factor",
            str(payload.error_boost_factor),
        ]
        ctx.run_command(
            args,
            cwd=self._repo_root,
            stage="Training statistical model",
            progress_range=(0.05, 1.0),
            on_output=handle_output,
        )

        if constraints_path.exists():
            ctx.add_artifact("constraints.json", constraints_path)
        if weights_path.exists():
            ctx.add_artifact("model_weights.json", weights_path)
        ctx.set_result(constraints=str(constraints_path), weights=str(weights_path))


__all__ = ["PipelineTasks"]
