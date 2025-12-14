import subprocess
import sys
from pathlib import Path

FIXTURES = Path("tests/fixtures")
EXPECTED_SRT = FIXTURES / "demo.expected.srt"


def _run_stage2(tmp_path: Path) -> Path:
    output_dir = tmp_path / "_inference_input"
    cmd = [
        sys.executable,
        "build_training_pair_standalone.py",
        "--primary-input",
        str(FIXTURES / "demo.txt"),
        "--asr-reference",
        str(FIXTURES / "demo.asr.visual.words.diar.json"),
        "--out-inference-dir",
        str(output_dir),
        "--config-file",
        str(FIXTURES / "pipeline_config.test.yaml"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output_dir / "demo.enriched.json"


def test_stage3_segments_expected_srt(tmp_path: Path) -> None:
    enriched = _run_stage2(tmp_path)

    output_dir = tmp_path / "output"
    output_srt = output_dir / "demo.srt"
    cmd = [
        sys.executable,
        "main.py",
        "--input",
        str(enriched),
        "--output",
        str(output_srt),
        "--config",
        str(FIXTURES / "config.test.yaml"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_srt.exists()

    expected_lines = EXPECTED_SRT.read_text(encoding="utf-8").strip().splitlines()
    actual_lines = output_srt.read_text(encoding="utf-8").strip().splitlines()
    assert actual_lines == expected_lines
