"""End-to-end smoke test for the ISCE pipeline using mock ASR data.

This script stitches together the three pipeline stages in a lightweight
configuration suitable for CI environments. It relies on a precomputed
mock ASR JSON to avoid WhisperX/model downloads.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.stderr.write(f"Command failed with code {result.returncode}: {' '.join(cmd)}\n")
        sys.exit(result.returncode)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        sys.stderr.write(f"Missing {description}: {path}\n")
        sys.exit(1)
    if path.is_file() and path.stat().st_size == 0:
        sys.stderr.write(f"Empty {description}: {path}\n")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mock end-to-end ISCE smoke test.")
    parser.add_argument("--workdir", type=Path, default=Path("tests/_artifacts"), help="Workspace for generated artifacts.")
    parser.add_argument("--media", type=Path, required=True, help="Path to the input media file.")
    parser.add_argument("--transcript", type=Path, required=True, help="Path to the edited transcript (TXT or SRT).")
    parser.add_argument("--mock-asr", type=Path, required=True, help="Path to the mock ASR JSON file.")
    parser.add_argument("--pipeline-config", type=Path, default=Path("pipeline_config.yaml"), help="Pipeline config YAML path.")
    parser.add_argument("--segmentation-config", type=Path, default=Path("config.yaml"), help="Segmentation config YAML path.")
    parser.add_argument("--output-dir", type=Path, help="Optional override for the SRT output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workdir = args.workdir
    align_dir = workdir / "_align"
    inference_dir = workdir / "_inference_input"
    output_dir = args.output_dir or (workdir / "output")

    for path in (workdir, align_dir, inference_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    media = args.media.resolve()
    transcript = args.transcript.resolve()
    mock_asr = args.mock_asr.resolve()
    pipeline_config = args.pipeline_config.resolve()
    segmentation_config = args.segmentation_config.resolve()

    base = media.stem

    # Stage 1: Mock ASR alignment
    align_cmd = [
        "python",
        "align_make.py",
        "--input-file",
        str(media),
        "--out-root",
        str(workdir),
        "--config-file",
        str(pipeline_config),
        "--mock-asr-json",
        str(mock_asr),
    ]
    run_command(align_cmd)

    align_output = align_dir / f"{base}.asr.visual.words.diar.json"
    require_file(align_output, "align output")

    # Stage 2: Build enriched tokens for inference
    build_cmd = [
        "python",
        "build_training_pair_standalone.py",
        "--primary-input",
        str(transcript),
        "--asr-reference",
        str(align_output),
        "--out-inference-dir",
        str(inference_dir),
        "--config-file",
        str(pipeline_config),
    ]
    run_command(build_cmd)

    enriched_output = inference_dir / f"{base}.enriched.json"
    require_file(enriched_output, "enriched inference tokens")

    # Stage 3: Segment and write SRT
    srt_output = output_dir / f"{base}.srt"
    segment_cmd = [
        "python",
        "main.py",
        "--input",
        str(enriched_output),
        "--output",
        str(srt_output),
        "--config",
        str(segmentation_config),
    ]
    run_command(segment_cmd)

    require_file(srt_output, "SRT output")
    print(f"[OK] Smoke test completed successfully. Output: {srt_output}")


if __name__ == "__main__":
    main()
