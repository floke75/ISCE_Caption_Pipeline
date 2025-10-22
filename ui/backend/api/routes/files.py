"""Filesystem browsing and validation endpoints for the UI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
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

    return router


__all__ = ["FileBrowser", "create_file_router"]
