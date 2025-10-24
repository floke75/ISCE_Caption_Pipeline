"""Interactive installer for the ISCE Caption Pipeline.

This script provisions a virtual environment, installs runtime
dependencies, and downloads the Swedish spaCy language model.
It aims to provide a turn-key experience so that users do not
need to orchestrate `pip` commands manually.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).parent.resolve()
DEFAULT_VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
DEV_REQUIREMENTS_FILE = REPO_ROOT / "requirements-dev.txt"


class InstallerError(RuntimeError):
    """Raised when an installation step fails."""


def run_command(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    """Run a subprocess command and stream its output."""

    print(f"\n→ Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise InstallerError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def ensure_python_version(min_major: int = 3, min_minor: int = 10) -> None:
    """Ensure the executing interpreter meets the minimum requirement."""

    if sys.version_info < (min_major, min_minor):
        raise InstallerError(
            "The installer requires Python >= "
            f"{min_major}.{min_minor}. Detected {platform.python_version()}"
        )


def prompt_gpu_backend(default_backend: str) -> str:
    """Prompt the user for their preferred spaCy GPU backend."""

    choices = {
        "1": "cpu",
        "2": "cuda12x",
        "3": "cuda11x",
        "4": "rocm",
    }

    default_key = next((key for key, value in choices.items() if value == default_backend), "1")

    print(
        "\nSelect the spaCy backend to install.\n"
        "  1) CPU (default)\n"
        "  2) CUDA 12.x (recommended for modern NVIDIA GPUs on Windows)\n"
        "  3) CUDA 11.x\n"
        "  4) ROCm (AMD GPUs on Linux)\n"
    )

    selection = input(f"Enter choice [1-4] (default: {default_key}): ").strip()
    return choices.get(selection, default_backend)


def resolve_backend(args: argparse.Namespace) -> str:
    """Determine which spaCy backend to install."""

    if args.gpu_backend != "auto":
        return args.gpu_backend

    if args.non_interactive:
        return "cpu"

    default_backend = "cuda12x" if platform.system() == "Windows" else "cpu"
    return prompt_gpu_backend(default_backend)


def create_virtualenv(venv_path: Path, *, force: bool) -> Path:
    """Create (or reuse) the virtual environment."""

    if venv_path.exists() and not force:
        print(f"Using existing virtual environment at {venv_path}.")
    else:
        if venv_path.exists() and force:
            print(f"Recreating virtual environment at {venv_path}.")
            shutil.rmtree(venv_path)
        print(f"Creating virtual environment at {venv_path}...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])

    python_path = venv_python_path(venv_path)
    if not python_path.exists():  # pragma: no cover - safety
        raise InstallerError(
            f"Virtual environment created at {venv_path}, but python was not found."
        )
    return python_path


def venv_python_path(venv_path: Path) -> Path:
    """Return the python executable inside the virtual environment."""

    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_requirements(
    python_path: Path,
    *,
    include_dev: bool,
    gpu_backend: str,
) -> None:
    """Install runtime (and optionally dev) dependencies."""

    pip_command = [str(python_path), "-m", "pip"]

    run_command(pip_command + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    run_command(pip_command + ["install", "-r", str(REQUIREMENTS_FILE)])

    if include_dev and DEV_REQUIREMENTS_FILE.exists():
        run_command(pip_command + ["install", "-r", str(DEV_REQUIREMENTS_FILE)])

    if gpu_backend != "cpu":
        run_command(pip_command + ["install", f"spacy[{gpu_backend}]"])

    run_command([str(python_path), "-m", "spacy", "download", "sv_core_news_lg"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Install the ISCE Caption Pipeline in an isolated environment.",
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV_DIR),
        help="Path to the virtual environment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu-backend",
        choices=["auto", "cpu", "cuda11x", "cuda12x", "rocm"],
        default="auto",
        help="spaCy GPU backend to install. Use 'auto' to be prompted.",
    )
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Also install development dependencies if requirements-dev.txt is present.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Recreate the virtual environment even if it already exists.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt for input; defaults will be used.",
    )
    return parser.parse_args(argv)


def summarize(python_path: Path) -> None:
    """Print final activation instructions."""

    venv_path = python_path.parent.parent
    if platform.system() == "Windows":
        activation = f"{venv_path}\\Scripts\\activate"
    else:
        activation = f"source {venv_path}/bin/activate"

    print(
        "\nInstallation complete!\n"
        "To activate the virtual environment run:\n"
        f"  {activation}\n\n"
        "Once activated you can run pipeline commands, for example:\n"
        "  python run_pipeline.py --help\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        ensure_python_version()
        venv_path = Path(args.venv).expanduser().resolve()
        gpu_backend = resolve_backend(args)
        python_path = create_virtualenv(venv_path, force=args.force_recreate)
        install_requirements(
            python_path,
            include_dev=args.include_dev,
            gpu_backend=gpu_backend,
        )
        summarize(python_path)
    except InstallerError as exc:
        print(f"\nInstallation failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        print("\nInstallation cancelled.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
