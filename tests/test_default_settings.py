from pathlib import Path

import align_make
import build_training_pair_standalone as build_pair
import run_pipeline
from pipeline_config import load_pipeline_config


TEST_KEYS = {
    "run_pipeline": [
        "project_root",
        "pipeline_root",
        "drop_folder_inference",
        "drop_folder_training",
        "srt_placement_folder",
        "txt_placement_folder",
        "processed_dir",
        "intermediate_dir",
        "output_dir",
    ],
    "align_make": [
        "project_root",
        "pipeline_root",
        ("align_make", "out_root"),
        ("align_make", "cache_dir"),
    ],
    "build_pair": [
        "project_root",
        "pipeline_root",
        ("build_pair", "in_txt_dir"),
        ("build_pair", "in_srt_dir"),
        ("build_pair", "out_training_dir"),
        ("build_pair", "out_inference_dir"),
    ],
}


def _assert_paths_under_root(cfg: dict, root: Path, keys: list[str | tuple[str, str]]):
    for key in keys:
        if isinstance(key, tuple):
            container, subkey = key
            value = cfg[container][subkey]
        else:
            value = cfg[key]
        assert str(value).startswith(str(root)), f"{key} not under {root}: {value!r}"


def test_default_settings_resolve_to_supplied_project_root(tmp_path):
    yaml_path = tmp_path / "pipeline_config.yaml"
    project_root = tmp_path / "project"
    yaml_path.write_text(f"project_root: '{project_root}'\n", encoding="utf-8")

    configs = {
        "run_pipeline": load_pipeline_config(run_pipeline.DEFAULT_SETTINGS, yaml_path=str(yaml_path)),
        "align_make": load_pipeline_config(align_make.DEFAULT_SETTINGS, yaml_path=str(yaml_path)),
        "build_pair": load_pipeline_config(build_pair.DEFAULT_SETTINGS, yaml_path=str(yaml_path)),
    }

    for name, cfg in configs.items():
        _assert_paths_under_root(cfg, project_root, TEST_KEYS[name])


def test_run_pipeline_load_configuration_honors_cli_override(tmp_path):
    config_file = tmp_path / "pipeline_config.yaml"
    project_root = tmp_path / "custom_root"
    config_file.write_text(f"project_root: '{project_root}'\n", encoding="utf-8")

    config = run_pipeline.load_configuration(config_file)

    assert Path(config["project_root"]) == project_root
    assert Path(config["pipeline_root"]) == project_root / "pipeline_data"
