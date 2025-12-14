from pathlib import Path
from types import SimpleNamespace

import ui.backend.pipelines as pipelines


class FakeContext:
    def __init__(self, workspace: Path, params: dict, runtime_config: dict):
        self.record = SimpleNamespace(params=params, workspace=workspace)
        self._runtime_config = runtime_config
        self._segmentation_config_path = runtime_config.get("segmentation_config_path")
        self.commands = []
        self.updates = []
        self.finalized = []

    def update(self, *, progress=None, message=None):
        self.updates.append((progress, message))

    def effective_config(self, overrides=None):
        overrides = overrides or {}
        merged = {**self._runtime_config, **overrides}
        return merged

    def segmentation_config(self, overrides=None):  # pragma: no cover - defensive fallback
        overrides = overrides or {}
        if self._segmentation_config_path:
            return Path(self._segmentation_config_path)
        raise RuntimeError("Segmentation config not available")

    def stream_command(self, command, cwd=None, env=None):
        cmd_list = list(command)
        self.commands.append(cmd_list)

        if "--out-root" in cmd_list:
            out_root = Path(cmd_list[cmd_list.index("--out-root") + 1])
            input_file = Path(cmd_list[cmd_list.index("--input-file") + 1])
            base_name = input_file.stem
            target = out_root / "_align" / f"{base_name}.asr.visual.words.diar.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")

        if "--out-inference-dir" in cmd_list:
            out_dir = Path(cmd_list[cmd_list.index("--out-inference-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            if "--output-basename" in cmd_list:
                base_name = cmd_list[cmd_list.index("--output-basename") + 1]
            else:
                primary_input = Path(cmd_list[cmd_list.index("--primary-input") + 1])
                base_name = primary_input.stem
            target = out_dir / f"{base_name}.enriched.json"
            target.write_text("{}", encoding="utf-8")

        if "--out-training-dir" in cmd_list:
            out_dir = Path(cmd_list[cmd_list.index("--out-training-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            primary_input = Path(cmd_list[cmd_list.index("--primary-input") + 1])
            target = out_dir / f"{primary_input.stem}.train.words.json"
            target.write_text("{}", encoding="utf-8")

        if "--output" in cmd_list:
            output_path = Path(cmd_list[cmd_list.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")

        if "train_model.py" in Path(cmd_list[1]).name:
            constraints_path = Path(cmd_list[cmd_list.index("--constraints") + 1])
            weights_path = Path(cmd_list[cmd_list.index("--weights") + 1])
            constraints_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            constraints_path.write_text("{}", encoding="utf-8")
            weights_path.write_text("{}", encoding="utf-8")

    def finalize(self, status, result=None, error=None):
        self.finalized.append((status, result, error))


def _runtime(tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    intermediate_dir = pipeline_root / "_intermediate"
    output_dir = pipeline_root / "_output"
    txt_dir = pipeline_root / "txt"
    pipeline_cfg = tmp_path / "pipeline_config.yaml"
    segmentation_cfg = tmp_path / "config.yaml"
    pipeline_cfg.write_text("pipeline: config", encoding="utf-8")
    segmentation_cfg.write_text("segmentation: config", encoding="utf-8")
    return {
        "__path__": pipeline_cfg,
        "pipeline_root": pipeline_root,
        "intermediate_dir": intermediate_dir,
        "output_dir": output_dir,
        "txt_placement_folder": txt_dir,
        "project_root": tmp_path,
        "segmentation_config_path": segmentation_cfg,
    }


def test_run_inference_commands_include_required_flags(tmp_path):
    media_path = tmp_path / "clip.mp4"
    transcript_path = tmp_path / "clip.txt"
    media_path.write_text("media", encoding="utf-8")
    transcript_path.write_text("transcript", encoding="utf-8")

    ctx = FakeContext(
        workspace=tmp_path / "workspace",
        params={"media_path": str(media_path), "transcript_path": str(transcript_path)},
        runtime_config=_runtime(tmp_path),
    )

    pipelines.run_inference(ctx)

    assert len(ctx.commands) == 3

    align_cmd = ctx.commands[0]
    expected_media_copy = ctx.record.workspace / "inputs" / "media" / media_path.name
    expected_out_root = Path(ctx._runtime_config["intermediate_dir"])
    expected_config = Path(ctx._runtime_config["__path__"])
    assert align_cmd[align_cmd.index("--input-file") + 1] == str(expected_media_copy)
    assert align_cmd[align_cmd.index("--out-root") + 1] == str(expected_out_root)
    assert align_cmd[align_cmd.index("--config-file") + 1] == str(expected_config)

    enrich_cmd = ctx.commands[1]
    expected_asr = expected_out_root / "_align" / f"{media_path.stem}.asr.visual.words.diar.json"
    expected_inference_dir = expected_out_root / "_inference_input"
    expected_transcript_copy = ctx.record.workspace / "inputs" / "text" / transcript_path.name
    assert "--primary-input" in enrich_cmd
    assert "--asr-reference" in enrich_cmd
    assert "--out-inference-dir" in enrich_cmd
    assert "--config-file" in enrich_cmd
    assert "--output-basename" in enrich_cmd
    assert "--asr-only-mode" not in enrich_cmd
    assert enrich_cmd[enrich_cmd.index("--primary-input") + 1] == str(expected_transcript_copy)
    assert enrich_cmd[enrich_cmd.index("--asr-reference") + 1] == str(expected_asr)
    assert enrich_cmd[enrich_cmd.index("--out-inference-dir") + 1] == str(expected_inference_dir)

    segment_cmd = ctx.commands[2]
    expected_output = Path(ctx._runtime_config["output_dir"]) / f"{media_path.stem}.srt"
    assert "main.py" in Path(segment_cmd[1]).name
    assert "--input" in segment_cmd and "--output" in segment_cmd and "--config" in segment_cmd
    assert segment_cmd[segment_cmd.index("--output") + 1] == str(expected_output)


def test_run_inference_adds_asr_only_flag(tmp_path):
    media_path = tmp_path / "clip.mp4"
    media_path.write_text("media", encoding="utf-8")

    ctx = FakeContext(
        workspace=tmp_path / "workspace_asr_only",
        params={"media_path": str(media_path)},
        runtime_config=_runtime(tmp_path),
    )

    pipelines.run_inference(ctx)

    enrich_cmd = ctx.commands[1]
    assert "--asr-only-mode" in enrich_cmd


def test_run_training_pair_commands_include_required_flags(tmp_path):
    media_path = tmp_path / "clip.mp4"
    srt_path = tmp_path / "clip.srt"
    media_path.write_text("media", encoding="utf-8")
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\ncaption", encoding="utf-8")

    ctx = FakeContext(
        workspace=tmp_path / "workspace_training",
        params={"media_path": str(media_path), "srt_path": str(srt_path)},
        runtime_config=_runtime(tmp_path),
    )

    pipelines.run_training_pair(ctx)

    assert len(ctx.commands) == 2

    align_cmd = ctx.commands[0]
    assert "--input-file" in align_cmd and "--out-root" in align_cmd and "--config-file" in align_cmd

    training_cmd = ctx.commands[1]
    assert "--primary-input" in training_cmd
    assert "--asr-reference" in training_cmd
    assert "--out-training-dir" in training_cmd
    assert "--config-file" in training_cmd


def test_run_model_training_commands_include_optional_args(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    ctx = FakeContext(
        workspace=tmp_path / "workspace_model_training",
        params={"corpus_dir": str(corpus_dir), "iterations": 2, "error_boost_factor": 1.5},
        runtime_config=_runtime(tmp_path),
    )

    pipelines.run_model_training(ctx)

    assert len(ctx.commands) == 1
    train_cmd = ctx.commands[0]
    assert "train_model.py" in Path(train_cmd[1]).name
    assert "--corpus" in train_cmd
    assert "--constraints" in train_cmd and "--weights" in train_cmd
    assert "--config" in train_cmd
    assert "--iterations" in train_cmd and "--error-boost-factor" in train_cmd

