"""Filesystem browsing and validation endpoints for the UI.

This module provides a secure and sandboxed way for the frontend to interact
with the server's file system. It defines a `FileBrowser` service that operates
only within a set of administrator-defined "root" directories, preventing
unauthorized file access.

The main components are:
-   **FileBrowser**: A service class that encapsulates all safe file system
    operations, such as listing directories and validating paths. It ensures
    that all paths are resolved and checked against the allowlisted roots.
-   **API Router**: The `create_file_router` function constructs and returns a
    FastAPI router with endpoints for:
    -   Listing the available root directories (`/roots`).
    -   Listing the contents of a directory (`/list`).
    -   Validating a given path (`/validate`), checking for existence, type
        (file/directory), and whether it is within an allowed root.
    -   **New**: Serving file content (`/content`) and downloads (`/download`).

These endpoints are crucial for the UI's file picker components and artifact
viewers, allowing users to select input/output paths and inspect results
safely.
"""
from __future__ import annotations
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class FileRoot:
    """Represents an allowlisted filesystem root."""

    id: str
    label: str
    path: Path


class FileRootModel(BaseModel):
    id: str
    label: str
    path: str


class BreadcrumbModel(BaseModel):
    label: str
    path: str


class FileEntryModel(BaseModel):
    name: str
    path: str
    is_dir: bool = Field(alias="isDir")
    is_file: bool = Field(alias="isFile")


class FileListingModel(BaseModel):
    root: FileRootModel
    path: str
    parent: Optional[str] = None
    breadcrumbs: List[BreadcrumbModel]
    entries: List[FileEntryModel]


class FileValidationModel(BaseModel):
    path: str
    exists: bool
    is_dir: bool = Field(alias="isDir")
    is_file: bool = Field(alias="isFile")
    allowed: bool
    root: Optional[FileRootModel] = None
    detail: Optional[str] = None


class FileContentModel(BaseModel):
    path: str
    content: str
    size: int
    mime_type: Optional[str] = Field(None, alias="mimeType")
    truncated: bool = False


class FileBrowser:
    """Performs safe filesystem operations within an allowlisted set of roots."""

    def __init__(self, roots: Iterable[Tuple[str, str, Path]]) -> None:
        normalized: Dict[str, FileRoot] = {}
        for identifier, label, path in roots:
            resolved = path.expanduser().resolve()
            normalized[identifier] = FileRoot(id=identifier, label=label, path=resolved)
        if not normalized:
            raise ValueError("At least one file root must be configured")
        self._roots = normalized
        self._root_order = list(normalized.keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_root_for_path(self, path: Path) -> Optional[FileRoot]:
        for root in self._roots.values():
            if self._is_within_root(path, root.path):
                return root
        return None

    @staticmethod
    def _is_within_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _normalize(self, raw: str) -> Path:
        return Path(raw).expanduser().resolve()

    def _ensure_allowed(self, raw: str) -> Tuple[Optional[FileRoot], Path]:
        path = self._normalize(raw)
        root = self._find_root_for_path(path)
        return root, path

    def _format_root_model(self, root: FileRoot) -> FileRootModel:
        return FileRootModel(id=root.id, label=root.label, path=str(root.path))

    def _breadcrumbs(self, root: FileRoot, path: Path) -> List[BreadcrumbModel]:
        crumbs: List[BreadcrumbModel] = []
        try:
            relative = path.relative_to(root.path)
        except ValueError:
            return crumbs
        current = root.path
        crumbs.append(BreadcrumbModel(label=root.label, path=str(root.path)))
        for part in relative.parts:
            current = current / part
            crumbs.append(BreadcrumbModel(label=part, path=str(current)))
        return crumbs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def roots(self) -> List[FileRootModel]:
        return [self._format_root_model(self._roots[key]) for key in self._root_order]

    def list_directory(self, *, path: Optional[str] = None, root_id: Optional[str] = None) -> FileListingModel:
        target_root: Optional[FileRoot] = None
        target_path: Optional[Path] = None

        if path:
            target_root, target_path = self._ensure_allowed(path)
            if target_root is None:
                raise HTTPException(status_code=403, detail="Path is outside the allowlisted directories")
        elif root_id:
            target_root = self._roots.get(root_id)
            if target_root is None:
                raise HTTPException(status_code=404, detail="Unknown root identifier")
            target_path = target_root.path
        else:
            default_root_id = self._root_order[0]
            target_root = self._roots[default_root_id]
            target_path = target_root.path

        assert target_root is not None
        assert target_path is not None

        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Directory does not exist")
        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        entries: List[FileEntryModel] = []
        try:
            children = list(target_path.iterdir())
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Permission denied") from exc

        for child in sorted(children, key=lambda item: (not item.is_dir(), item.name.lower())):
            entries.append(
                FileEntryModel(
                    name=child.name,
                    path=str(child),
                    isDir=child.is_dir(),
                    isFile=child.is_file(),
                )
            )

        parent: Optional[str] = None
        if target_path != target_root.path:
            parent_candidate = target_path.parent
            if self._is_within_root(parent_candidate, target_root.path):
                parent = str(parent_candidate)

        return FileListingModel(
            root=self._format_root_model(target_root),
            path=str(target_path),
            parent=parent,
            breadcrumbs=self._breadcrumbs(target_root, target_path),
            entries=entries,
        )

    def validate_path(self, path: str) -> FileValidationModel:
        root, resolved = self._ensure_allowed(path)
        exists = resolved.exists()
        is_dir = exists and resolved.is_dir()
        is_file = exists and resolved.is_file()
        allowed = root is not None
        detail: Optional[str] = None
        if not allowed:
            detail = "Path is outside the allowlisted directories"
        elif not exists:
            detail = "Path does not exist"

        root_model = self._format_root_model(root) if root else None
        return FileValidationModel(
            path=str(resolved),
            exists=exists,
            isDir=is_dir,
            isFile=is_file,
            allowed=allowed,
            root=root_model,
            detail=detail,
        )

    def get_file_path_secure(self, path: str) -> Path:
        """Validates that a path is safe, exists, and is a file."""
        root, resolved = self._ensure_allowed(path)
        if root is None:
            raise HTTPException(status_code=403, detail="Path is outside the allowlisted directories")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="File does not exist")
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        return resolved

    def read_text_content(self, path: str, limit: int = 500_000) -> FileContentModel:
        """Reads the text content of a file, respecting a byte limit."""
        resolved = self.get_file_path_secure(path)
        size = resolved.stat().st_size
        mime, _ = mimetypes.guess_type(resolved)

        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}") from exc

        truncated = size > len(content.encode("utf-8"))
        return FileContentModel(
            path=str(resolved),
            content=content,
            size=size,
            mimeType=mime or "text/plain",
            truncated=truncated,
        )


def create_file_router(browser: FileBrowser) -> APIRouter:
    """Create a router exposing filesystem helpers."""

    router = APIRouter(prefix="/api/files", tags=["files"])

    @router.get("/roots", response_model=List[FileRootModel])
    def get_roots() -> List[FileRootModel]:
        return browser.roots()

    @router.get("/list", response_model=FileListingModel)
    def list_directory(
        path: Optional[str] = Query(None, description="Absolute path of the directory to list"),
        root: Optional[str] = Query(None, description="Identifier of the allowlisted root to list when no path is provided"),
    ) -> FileListingModel:
        return browser.list_directory(path=path, root_id=root)

    @router.get("/validate", response_model=FileValidationModel)
    def validate_path(path: str = Query(..., description="Absolute path to validate")) -> FileValidationModel:
        return browser.validate_path(path)

    @router.get("/content", response_model=FileContentModel)
    def get_file_content(
        path: str = Query(..., description="Absolute path to the file"),
        limit: int = Query(500_000, description="Max bytes to read"),
    ) -> FileContentModel:
        """Reads the content of a file as text (max 500KB by default)."""
        return browser.read_text_content(path, limit=limit)

    @router.get("/download")
    def download_file(path: str = Query(..., description="Absolute path to download")) -> FileResponse:
        """Downloads a file as an attachment."""
        resolved = browser.get_file_path_secure(path)
        return FileResponse(resolved, media_type="application/octet-stream", filename=resolved.name)

    return router


__all__ = ["FileBrowser", "create_file_router"]
