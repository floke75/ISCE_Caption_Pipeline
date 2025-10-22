"""Utilities to translate API requests into executable pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .job_manager import JobStep


def build_inference_steps(workspace: Path, config: Dict[str, object], params: Dict[str, object]) -> Iterable[JobStep]:
    media_path = Path(params["mediaPath"])
    transcript_path = Path(params["transcriptPath"])
    config_path = workspace / "pipeline_config.yaml"
    intermediate_root = Path(config["align_make"]["out_root"])  # type: ignore[index]
    inference_dir = Path(config["build_pair"]["out_inference_dir"])  # type: ignore[index]
    enriched_path = inference_dir / f"{media_path.stem}.enriched.json"
    srt_output = workspace / "output" / f"{media_path.stem}.srt"
    srt_output.parent.mkdir(parents=True, exist_ok=True)

    python = params.get("pythonExecutable", "python")
    project_root = Path(__file__).resolve().parents[2]

    segmentation_config = project_root / "config.yaml"

    steps: List[JobStep] = [
        JobStep(
            name="Generate ASR reference",
            command=[
                python,
                str(project_root / "align_make.py"),
                "--input-file",
                str(media_path),
                "--out-root",
                str(intermediate_root),
                "--config-file",
                str(config_path),
            ],
            cwd=project_root,
        ),
        JobStep(
            name="Align transcript and enrich tokens",
            command=[
                python,
                str(project_root / "build_training_pair_standalone.py"),
                "--primary-input",
                str(transcript_path),
                "--asr-reference",
                str((intermediate_root / "_align" / f"{media_path.stem}.asr.visual.words.diar.json")),
                "--config-file",
                str(config_path),
                "--out-inference-dir",
                str(inference_dir),
                "--output-basename",
                media_path.stem,
            ],
            cwd=project_root,
        ),
        JobStep(
            name="Run segmentation",
            command=[
                python,
                str(project_root / "main.py"),
                "--input",
                str(enriched_path),
                "--output",
                str(srt_output),
                "--config",
                str(segmentation_config),
            ],
            cwd=project_root,
        ),
    ]
    return steps


def build_training_pair_steps(workspace: Path, config: Dict[str, object], params: Dict[str, object]) -> Iterable[JobStep]:
    transcript_path = Path(params["transcriptPath"])
    asr_reference = Path(params["asrReference"])
    config_path = workspace / "pipeline_config.yaml"
    mode = params.get("mode", "inference")
    build_cfg = config["build_pair"]  # type: ignore[index]
    inference_dir = Path(build_cfg["out_inference_dir"])  # type: ignore[index]
    training_dir = Path(build_cfg["out_training_dir"])  # type: ignore[index]
    inference_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    python = params.get("pythonExecutable", "python")
    project_root = Path(__file__).resolve().parents[2]

    basename = params.get("outputBasename") or transcript_path.stem

    return [
        JobStep(
            name="Build training pair",
            command=[
                python,
                str(project_root / "build_training_pair_standalone.py"),
                "--primary-input",
                str(transcript_path),
                "--asr-reference",
                str(asr_reference),
                "--config-file",
                str(config_path),
                "--out-inference-dir",
                str(inference_dir),
                "--out-training-dir",
                str(training_dir),
                "--output-basename",
                str(basename),
            ],
            cwd=project_root,
        )
    ]


def build_training_steps(workspace: Path, config: Dict[str, object], params: Dict[str, object]) -> Iterable[JobStep]:
    python = params.get("pythonExecutable", "python")
    project_root = Path(__file__).resolve().parents[2]
    corpus_dir = Path(params["corpusDir"])
    iterations = str(params.get("iterations", 3))
    error_boost = str(params.get("errorBoostFactor", 1.0))

    output_dir = workspace / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    constraints_path = output_dir / "constraints.json"
    weights_path = output_dir / "model_weights.json"
    config_path = workspace / "pipeline_config.yaml"

    return [
        JobStep(
            name="Train statistical model",
            command=[
                python,
                str(project_root / "scripts" / "train_model.py"),
                "--corpus",
                str(corpus_dir),
                "--constraints",
                str(constraints_path),
                "--weights",
                str(weights_path),
                "--config",
                str(config_path),
                "--iterations",
                iterations,
                "--error-boost-factor",
                error_boost,
            ],
            cwd=project_root,
        )
    ]


__all__ = [
    "build_inference_steps",
    "build_training_pair_steps",
    "build_training_steps",
]
