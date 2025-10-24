#!/usr/bin/env python3
"""One-click installer for the ISCE Captioning Pipeline.

This script provisions a virtual environment, installs Python requirements,
downloads the Swedish SpaCy model, and hydrates the frontend dependencies. It
is intended to offer a guided, cross-platform installation experience so users
do not have to juggle environment management or package commands manually.

Usage (from the repository root):

    python scripts/install.py

Use ``--help`` to discover optional flags such as GPU-enabled SpaCy installs or
frontend skipping.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List
import venv


MIN_PYTHON = (3, 11)
SPACY_MODEL = "sv_core_news_lg"


class InstallationError(RuntimeError):
    """Raised when a subprocess exits with a non-zero return code."""


def debug(msg: str) -> None:
    """Emit a human-friendly status message."""

    print(f"\n▶ {msg}")


def run_command(command: Iterable[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    """Execute a subprocess, surfacing failures as actionable errors."""

    pretty = " ".join(command)
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise InstallationError(f"Command failed ({result.returncode}): {pretty}")


def ensure_python_version() -> None:
    """Validate the interpreter version running the installer."""

    if sys.version_info < MIN_PYTHON:
        minimum = ".".join(map(str, MIN_PYTHON))
        current = platform.python_version()
        raise InstallationError(
            f"Python {minimum} or higher is required. Current interpreter: {current}."
        )


def build_virtualenv(venv_path: Path, *, recreate: bool) -> None:
    """Create (or recreate) the virtual environment."""

    if venv_path.exists() and recreate:
        debug(f"Removing existing virtual environment at {venv_path}")
        shutil.rmtree(venv_path)

    if not venv_path.exists():
        debug(f"Creating virtual environment at {venv_path}")
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_path)
    else:
        debug(f"Using existing virtual environment at {venv_path}")


def venv_bin(venv_path: Path, executable: str) -> Path:
    """Return the path to an executable within the virtual environment."""

    if os.name == "nt":
        candidate = venv_path / "Scripts" / f"{executable}.exe"
    else:
        candidate = venv_path / "bin" / executable

    if not candidate.exists():
        raise InstallationError(
            f"Expected executable {executable!r} inside the virtual environment was not found at {candidate}."
        )
    return candidate


def pip_install(pip_executable: Path, packages: Iterable[str]) -> None:
    """Install one or more packages via pip."""

    run_command([str(pip_executable), "install", "--upgrade", *packages])


def pip_install_requirements(pip_executable: Path, requirements_file: Path) -> None:
    """Install packages from requirements.txt."""

    if not requirements_file.exists():
        raise InstallationError(f"Requirements file not found: {requirements_file}")
    run_command([str(pip_executable), "install", "-r", str(requirements_file)])


def install_spacy_model(python_executable: Path, *, gpu: bool) -> None:
    """Ensure SpaCy and the Swedish language model are installed."""

    extras: List[str] = []
    if gpu:
        extras.append("spacy[cuda12x]")
    else:
        extras.append("spacy")

    # Upgrade SpaCy with the chosen extras (no-op if already installed).
    run_command([str(python_executable), "-m", "pip", "install", "--upgrade", *extras])

    # Download the language model via the module to avoid relying on static URLs.
    run_command([str(python_executable), "-m", "spacy", "download", SPACY_MODEL])


def install_frontend_dependencies(project_root: Path, *, skip_frontend: bool, warnings: List[str]) -> None:
    """Run npm install for the React frontend if npm is available."""

    if skip_frontend:
        warnings.append("Skipped frontend dependency installation as requested.")
        return

    npm_path = shutil.which("npm")
    frontend_dir = project_root / "ui" / "frontend"
    if npm_path is None:
        warnings.append(
            "npm was not found on PATH; skipped frontend dependency installation. "
            "Install Node.js from https://nodejs.org/ and re-run the installer to enable the web UI."
        )
        return

    if not frontend_dir.exists():
        warnings.append(f"Frontend directory not found at {frontend_dir}, skipping npm install.")
        return

    debug("Installing frontend dependencies via npm")
    run_command([npm_path, "install"], cwd=frontend_dir)


def check_ffmpeg(warnings: List[str]) -> None:
    """Ensure ffmpeg is discoverable, warning (but not failing) if missing."""

    if shutil.which("ffmpeg") is None:
        warnings.append(
            "ffmpeg was not detected on PATH. Install it from https://ffmpeg.org/ and ensure it is available "
            "before running the pipeline."
        )


def summarize(warnings: List[str], venv_path: Path) -> None:
    """Print a human-friendly summary of the installation outcome."""

    message = textwrap.dedent(
        f"""
        ✅ ISCE installation completed successfully.

        Next steps:
          • Activate the virtual environment located at: {venv_path}
            - Windows: {venv_path}\\Scripts\\activate
            - macOS/Linux: source {venv_path}/bin/activate
          • Launch the pipeline via `python run_pipeline.py` or start the web UI backend with
            `uvicorn ui.backend.app:app --host 0.0.0.0 --port 8000`.
          • For the frontend, run `npm run dev` inside ui/frontend (if npm dependencies were installed).
        """
    ).strip()
    print(f"\n{message}\n")

    if warnings:
        print("⚠️  The installer completed with warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the ISCE Captioning Pipeline dependencies.")
    parser.add_argument(
        "--venv",
        type=Path,
        default=Path(".venv"),
        help="Location for the Python virtual environment (default: .venv)",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Delete and recreate the virtual environment before installing.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Install GPU-enabled SpaCy dependencies (uses spacy[cuda12x]).",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip npm install for the React frontend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    warnings: List[str] = []

    ensure_python_version()

    build_virtualenv(args.venv, recreate=args.recreate_venv)

    pip_executable = venv_bin(args.venv, "pip")
    python_executable = venv_bin(args.venv, "python")

    debug("Upgrading pip, setuptools, and wheel")
    pip_install(pip_executable, ["pip", "setuptools", "wheel"])

    debug("Installing Python requirements")
    pip_install_requirements(pip_executable, project_root / "requirements.txt")

    debug("Installing SpaCy and downloading the Swedish language model")
    install_spacy_model(python_executable, gpu=args.gpu)

    debug("Verifying Python dependency graph with pip check")
    try:
        run_command([str(pip_executable), "check"])
    except InstallationError as error:
        warnings.append(str(error))

    install_frontend_dependencies(project_root, skip_frontend=args.skip_frontend, warnings=warnings)
    check_ffmpeg(warnings)

    summarize(warnings, args.venv)


if __name__ == "__main__":
    try:
        main()
    except InstallationError as exc:
        print(f"\n❌ Installation aborted: {exc}")
        sys.exit(1)
