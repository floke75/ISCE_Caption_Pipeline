"""Smoke tests for the CLI entrypoint in ``main.py``.

These tests focus on ensuring the command-line surface stays wired up even
when the heavy dependencies are exercised via the guided installer.  They are
intentionally lightweight so that CI retains coverage of the user-facing
interface without requiring the large statistical model artefacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import importlib

import pytest


@pytest.fixture(autouse=True)
def restore_argv():
    original = sys.argv[:]
    try:
        yield
    finally:
        sys.argv = original


def test_main_requires_input_arguments():
    """Invoking ``main.main`` without the mandatory flags exits gracefully."""
    sys.argv = ["main"]
    with pytest.raises(SystemExit):
        import main as main_module

        main_module.main()


def test_main_reports_missing_files(tmp_path: Path):
    """The CLI surfaces a helpful error when the input file is absent."""
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")

    output = tmp_path / "output.srt"

    sys.argv = [
        "main",
        "--input",
        str(tmp_path / "missing.json"),
        "--output",
        str(output),
        "--config",
        str(config),
    ]

    with pytest.raises(SystemExit):
        main_module = importlib.import_module("main")
        main_module.main()
