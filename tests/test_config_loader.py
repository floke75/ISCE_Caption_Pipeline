from pathlib import Path

import yaml

from pipeline_config import load_pipeline_config


def test_load_pipeline_config_resolves_placeholders(tmp_path: Path) -> None:
    defaults = {
        "project_root": str(tmp_path / "project"),
        "pipeline_root": "{project_root}/pipeline_data",
        "align_make": {"cache_dir": "{pipeline_root}/_cache", "batch_size": 1},
        "build_pair": {"out_inference_dir": "{pipeline_root}/_inference_input"},
    }

    cfg = load_pipeline_config(defaults, yaml_path=str(tmp_path / "missing.yaml"))

    assert cfg["pipeline_root"].endswith("pipeline_data")
    assert cfg["align_make"]["cache_dir"].endswith("pipeline_data/_cache")
    assert cfg["build_pair"]["out_inference_dir"].endswith("pipeline_data/_inference_input")


def test_load_pipeline_config_merges_overrides(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pipeline_config.yaml"
    override = {
        "project_root": str(tmp_path),
        "pipeline_root": "{project_root}/custom_root",
        "align_make": {"batch_size": 4, "language": "en"},
    }
    yaml_path.write_text(yaml.safe_dump(override), encoding="utf-8")

    cfg = load_pipeline_config({"align_make": {"batch_size": 1}}, yaml_path=str(yaml_path))

    assert cfg["align_make"]["batch_size"] == 4
    assert cfg["align_make"]["language"] == "en"
    assert cfg["pipeline_root"].endswith("custom_root")
