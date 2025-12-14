import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path("tests/fixtures")


def test_stage2_generates_enriched_tokens(tmp_path: Path) -> None:
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

    enriched = output_dir / "demo.enriched.json"
    assert enriched.exists(), "Stage 2 should produce an enriched JSON file"

    payload = json.loads(enriched.read_text(encoding="utf-8"))
    words = [token["w"] for token in payload["tokens"]]
    assert words == ["Hello", "world", "demo", "clip."]
