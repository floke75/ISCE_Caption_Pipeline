#!/usr/bin/env python3
"""One-click installer for the ISCE captioning pipeline.

This script automates creation of a virtual environment, installs all Python
dependencies (optionally including the GPU-enabled SpaCy build), and downloads
the Swedish SpaCy language model required by the pipeline. It is safe to run
multiple times; subsequent runs will reuse the existing environment unless the
``--force-recreate`` flag is supplied.

Examples
--------

CPU-only install using the default ``.venv`` folder::

    python scripts/install_pipeline.py

Windows install with CUDA-accelerated SpaCy::

    python scripts/install_pipeline.py --gpu

Install into the active interpreter without creating a virtual environment::

    python scripts/install_pipeline.py --no-venv
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_VENV_DIR = Path(".venv")
DEFAULT_REQUIREMENTS = Path("requirements.txt")
DEFAULT_SPACY_MODEL = "sv_core_news_lg"
CUDA_VARIANTS = {"cuda12x"}


class InstallError(RuntimeError):
    """Raised when a subprocess returns a non-zero exit code."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install all Python dependencies for the ISCE captioning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run this script from the repository root for best results.",
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=DEFAULT_VENV_DIR,
        help="Where to create the virtual environment (default: .venv).",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Install into the currently active interpreter instead of creating a virtual environment.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete and recreate the virtual environment if it already exists.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Path to the requirements.txt file (default: repository root requirements.txt).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Install the CUDA-enabled SpaCy build (currently spaCy[cuda12x]).",
    )
    parser.add_argument(
        "--cuda-variant",
        default="cuda12x",
        choices=sorted(CUDA_VARIANTS),
        help="SpaCy CUDA variant to install when --gpu is supplied (default: cuda12x).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SPACY_MODEL,
        help="SpaCy language model to download (default: sv_core_news_lg).",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip downloading the SpaCy language model.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically answer yes to all confirmation prompts.",
    )
    return parser.parse_args(argv)


def _run_command(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    pretty = " ".join(_maybe_quote(part) for part in command)
    print(f"\n→ Running: {pretty}")
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise InstallError(f"Command failed with exit code {exc.returncode}: {pretty}") from exc


def _maybe_quote(token: str) -> str:
    if " " in token or "\t" in token:
        return f'"{token}"'
    return token


def _prompt_yes_no(message: str, *, default: bool, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    if not sys.stdin.isatty():
        return default

    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{message} {suffix} ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please respond with 'y' or 'n'.")


def _ensure_virtualenv(
    env_dir: Path,
    *,
    recreate: bool,
    auto_yes: bool,
) -> Path:
    env_dir = env_dir.expanduser().resolve()
    python_in_env = env_dir / ("Scripts" if platform.system() == "Windows" else "bin") / "python"

    if env_dir.exists():
        if recreate:
            print(f"Removing existing virtual environment at {env_dir}...")
            shutil.rmtree(env_dir)
        else:
            reuse = _prompt_yes_no(
                f"A virtual environment already exists at {env_dir}. Reuse it?",
                default=True,
                auto_yes=auto_yes,
            )
            if reuse:
                print(f"Using existing virtual environment at {env_dir}.")
                return python_in_env
            print(f"Deleting {env_dir} so a fresh environment can be created...")
            shutil.rmtree(env_dir)

    parent = env_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating virtual environment in {env_dir}...")
    _run_command([sys.executable, "-m", "venv", str(env_dir)])
    return python_in_env


def _pip_install(python_exe: Path, packages: Iterable[str]) -> None:
    cmd = [str(python_exe), "-m", "pip", "install", *packages]
    _run_command(cmd)


def _pip_install_requirements(python_exe: Path, requirements_path: Path) -> None:
    cmd = [str(python_exe), "-m", "pip", "install", "-r", str(requirements_path)]
    _run_command(cmd)


def _pip_check(python_exe: Path) -> None:
    _run_command([str(python_exe), "-m", "pip", "check"])


def _download_spacy_model(python_exe: Path, model: str) -> None:
    _run_command([str(python_exe), "-m", "spacy", "download", model])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    requirements = (repo_root / args.requirements).resolve()
    if not requirements.exists():
        raise InstallError(f"Requirements file not found: {requirements}")

    if args.no_venv:
        python_exe = Path(sys.executable)
        env_message = "Installing into the active interpreter." \
            "  (Use --venv-path to isolate dependencies.)"
        print(env_message)
    else:
        python_exe = _ensure_virtualenv(
            args.venv_path,
            recreate=args.force_recreate,
            auto_yes=args.yes,
        )

    print("\nUpgrading pip, setuptools, and wheel...")
    _pip_install(python_exe, ["--upgrade", "pip", "setuptools", "wheel"])

    print("\nInstalling core requirements...")
    _pip_install_requirements(python_exe, requirements)

    if args.gpu:
        if platform.system() != "Windows":
            warn = (
                "GPU mode was requested on a non-Windows platform. spaCy[cuda12x] "
                "is primarily distributed for Windows and may not install correctly."
            )
            proceed = _prompt_yes_no(warn + " Continue anyway?", default=False, auto_yes=args.yes)
            if not proceed:
                print("Skipping GPU install; continuing with CPU-only dependencies.")
            else:
                print("\nInstalling SpaCy with CUDA support...")
                _pip_install(python_exe, ["--upgrade", f"spacy[{args.cuda_variant}]"])
        else:
            print("\nInstalling SpaCy with CUDA support...")
            _pip_install(python_exe, ["--upgrade", f"spacy[{args.cuda_variant}]"])

    if not args.skip_model_download:
        print(f"\nDownloading SpaCy model '{args.model}'...")
        _download_spacy_model(python_exe, args.model)
    else:
        print("Skipping SpaCy model download.")

    print("\nVerifying installed packages...")
    _pip_check(python_exe)

    if not args.no_venv:
        activation_hint = _activation_hint(args.venv_path)
        print("\nInstallation complete! Activate your environment with:")
        print(f"  {activation_hint}")
    else:
        print("\nInstallation complete! (Dependencies installed into the active interpreter.)")

    print("Run the pipeline with: python run_pipeline.py")
    return 0


def _activation_hint(env_dir: Path) -> str:
    env_dir = env_dir.expanduser().resolve()
    if platform.system() == "Windows":
        return fr"{env_dir}\Scripts\activate"
    return f"source {env_dir}/bin/activate"


if __name__ == "__main__":  # pragma: no cover - script entry point
    try:
        raise SystemExit(main())
    except InstallError as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)
