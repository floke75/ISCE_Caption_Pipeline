"""Interactive installer for the ISCE Caption Pipeline.

The script automates virtual environment creation, dependency installation,
and spaCy model provisioning. It is designed to be idempotent and safe to run
multiple times, recreating the environment on demand.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_VENV = Path(".venv")
REQUIREMENTS_FILE = Path("requirements.txt")
SPACY_MODEL = "sv_core_news_lg"


class InstallationError(RuntimeError):
    """Raised when a subprocess invoked by the installer fails."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    venv_path = args.venv_path.resolve()
    ensure_python_version()

    if args.recreate and venv_path.exists():
        print(f"[installer] Removing existing environment at {venv_path} ...")
        shutil.rmtree(venv_path)

    python_path = create_virtualenv(venv_path)
    pip = [str(python_path), "-m", "pip"]

    run_command(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"],
                "Upgrading packaging tooling")

    if REQUIREMENTS_FILE.exists():
        run_command(pip + ["install", "-r", str(REQUIREMENTS_FILE)],
                    "Installing core requirements")
    else:
        print(f"[installer] Skipping requirements installation; {REQUIREMENTS_FILE} not found.")

    install_spacy(pip, args.spacy_accelerator)
    download_spacy_model(python_path)

    print("\n[installer] Installation completed successfully!")
    print(f"[installer] Activate the environment with: {activation_hint(venv_path)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up a virtual environment and install all pipeline dependencies.",
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=DEFAULT_VENV,
        help="Location for the virtual environment (default: .venv).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the virtual environment even if it already exists.",
    )
    parser.add_argument(
        "--spacy-accelerator",
        choices=["cpu", "cuda12x"],
        default="cpu",
        help=(
            "Choose the spaCy build to install. Use 'cuda12x' for NVIDIA GPU acceleration "
            "(Windows/Linux, CUDA 12.x)."
        ),
    )
    return parser


def ensure_python_version() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit(
            "Python 3.11 or newer is required to run the installer. "
            f"Detected {platform.python_version()}."
        )


def create_virtualenv(venv_path: Path) -> Path:
    """Create or reuse a virtual environment and return its Python executable."""

    if not venv_path.exists():
        print(f"[installer] Creating virtual environment at {venv_path} ...")
        run_command([sys.executable, "-m", "venv", str(venv_path)],
                    "Creating virtual environment")
    else:
        print(f"[installer] Using existing virtual environment at {venv_path}.")

    python_path = venv_python_path(venv_path)
    if not python_path.exists():
        raise InstallationError(
            f"Unable to locate Python executable inside the virtual environment at {python_path}."
        )
    return python_path


def install_spacy(pip: List[str], accelerator: str) -> None:
    package = "spacy" if accelerator == "cpu" else f"spacy[{accelerator}]"
    description = "Installing spaCy (CPU build)" if accelerator == "cpu" else "Installing spaCy with CUDA support"
    run_command(pip + ["install", "-U", package], description)


def download_spacy_model(python_path: Path) -> None:
    run_command(
        [str(python_path), "-m", "spacy", "download", SPACY_MODEL],
        f"Downloading spaCy model '{SPACY_MODEL}'",
    )


def activation_hint(venv_path: Path) -> str:
    if os.name == "nt":
        return f"{venv_path}\\Scripts\\activate"
    return f"source {venv_path}/bin/activate"


def venv_python_path(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run_command(command: Iterable[str], description: str) -> None:
    printable_command = " ".join(command)
    print(f"[installer] {description} ...")
    print(f"           $ {printable_command}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise InstallationError(
            f"Command failed with exit code {exc.returncode}: {printable_command}"
        ) from exc


if __name__ == "__main__":
    try:
        main()
    except InstallationError as error:
        print(f"\n[installer] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
