from __future__ import annotations

"""Utilities for validating user-supplied filesystem paths."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


PathKind = Literal["file", "directory", "any"]


@dataclass(frozen=True)
class PathStatus:
    """Result of validating a filesystem path."""

    path: Path
    exists: bool
    is_file: bool
    is_dir: bool
    root: Path


class PathValidationError(ValueError):
    """Raised when a supplied path cannot be safely used."""


def _build_allowlist() -> list[Path]:
    """Return a list of directories that user inputs may target."""

    repo_root = Path(__file__).resolve().parent.parent
    runtime_root = repo_root / "ui_runtime"
    workspace_root = Path("/workspace")
    home = Path.home()

    env_roots = os.environ.get("PIPELINE_ALLOWED_ROOTS")
    additional: tuple[Path, ...] = ()
    if env_roots:
        additional = tuple(Path(item.strip()) for item in env_roots.split(os.pathsep) if item.strip())

    candidates: Iterable[Path] = (
        repo_root,
        runtime_root,
        workspace_root,
        home,
        *additional,
    )

    allowlist: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except RuntimeError:
            # Some environments cannot resolve non-existent parents. Skip them.
            continue
        if resolved not in allowlist:
            allowlist.append(resolved)
    return allowlist


ALLOWED_ROOTS: list[Path] = _build_allowlist()


def describe_allowlist() -> list[str]:
    """Expose the allowlist as strings for API consumers."""

    return [str(root) for root in ALLOWED_ROOTS]


def _ensure_within_allowlist(path: Path) -> Path:
    """Ensure the resolved path is nested within an allowed root."""

    for root in ALLOWED_ROOTS:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return root
    allowed = ", ".join(describe_allowlist()) or "<none>"
    raise PathValidationError(
        f"Path {path} is outside the allowed roots. Allowed locations: {allowed}"
    )


def _normalise(path: str) -> Path:
    """Expand environment markers and resolve the given path."""

    raw = path.strip()
    if not raw:
        raise PathValidationError("Path cannot be blank")
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise PathValidationError("Paths must be absolute")
    return expanded.resolve(strict=False)


def _ensure_parent_exists(path: Path, description: str) -> None:
    parent = path.parent
    if not parent.exists():
        raise PathValidationError(
            f"{description} parent directory does not exist: {parent}"
        )
    if not parent.is_dir():
        raise PathValidationError(
            f"{description} parent path is not a directory: {parent}"
        )


def validate_path(
    raw_path: str,
    *,
    kind: PathKind,
    must_exist: bool = True,
    allow_create: bool = False,
    purpose: str | None = None,
) -> PathStatus:
    """Validate a path and return its resolved representation."""

    description = purpose or "Path"
    resolved = _normalise(raw_path)
    root = _ensure_within_allowlist(resolved)

    exists = resolved.exists()
    is_file = resolved.is_file()
    is_dir = resolved.is_dir()

    if must_exist and not exists:
        raise PathValidationError(f"{description} does not exist: {resolved}")

    if kind == "file":
        if exists and not is_file:
            raise PathValidationError(f"{description} must be a file: {resolved}")
        if not exists:
            if not allow_create:
                raise PathValidationError(f"{description} does not exist: {resolved}")
            _ensure_parent_exists(resolved, description)
    elif kind == "directory":
        if exists and not is_dir:
            raise PathValidationError(f"{description} must be a directory: {resolved}")
        if not exists:
            if not allow_create:
                raise PathValidationError(f"{description} does not exist: {resolved}")
            _ensure_parent_exists(resolved, description)

    else:  # kind == "any"
        if exists and not (is_file or is_dir):
            raise PathValidationError(f"{description} must refer to a file or directory")
        if not exists and not allow_create and must_exist:
            raise PathValidationError(f"{description} does not exist: {resolved}")
        if not exists and allow_create:
            _ensure_parent_exists(resolved, description)

    return PathStatus(
        path=resolved,
        exists=exists,
        is_file=is_file,
        is_dir=is_dir,
        root=root,
    )


def require_path(
    raw_path: str,
    *,
    kind: PathKind,
    must_exist: bool = True,
    allow_create: bool = False,
    purpose: str | None = None,
) -> Path:
    """Validate the path and return it as a ``Path`` object, or raise."""

    status = validate_path(
        raw_path,
        kind=kind,
        must_exist=must_exist,
        allow_create=allow_create,
        purpose=purpose,
    )
    return status.path
