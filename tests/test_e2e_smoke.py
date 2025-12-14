import subprocess
import sys
from pathlib import Path

FIXTURES = Path("tests/fixtures")
EXPECTED_SRT = FIXTURES / "demo.expected.srt"


def test_smoke_e2e_script_runs(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/smoke_e2e.py",
        "--workdir",
        str(tmp_path),
        "--media",
        str(FIXTURES / "demo.mp4"),
        "--transcript",
        str(FIXTURES / "demo.txt"),
        "--mock-asr",
        str(FIXTURES / "demo.asr.visual.words.diar.json"),
        "--pipeline-config",
        str(FIXTURES / "pipeline_config.test.yaml"),
        "--segmentation-config",
        str(FIXTURES / "config.test.yaml"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    srt_output = tmp_path / "output" / "demo.srt"
    assert srt_output.exists()

    expected_lines = EXPECTED_SRT.read_text(encoding="utf-8").strip().splitlines()
    actual_lines = srt_output.read_text(encoding="utf-8").strip().splitlines()
    assert actual_lines == expected_lines
