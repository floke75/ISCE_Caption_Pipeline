from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui.backend.api.routes.files import FileBrowser, create_file_router


@pytest.fixture()
def file_browser_app(tmp_path: Path) -> Iterator[Tuple[TestClient, Path, Path]]:
    root_a = tmp_path / "alpha"
    root_b = tmp_path / "beta"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "example.txt").write_text("hello", encoding="utf-8")
    nested = root_a / "nested"
    nested.mkdir()

    browser = FileBrowser(
        [
            ("alpha", "Alpha root", root_a),
            ("beta", "Beta root", root_b),
        ]
    )

    app = FastAPI()
    app.include_router(create_file_router(browser))
    client = TestClient(app)

    yield client, root_a, nested


def test_roots_endpoint_returns_allowlist(file_browser_app: Tuple[TestClient, Path, Path]) -> None:
    client, root_a, _nested = file_browser_app
    response = client.get("/api/files/roots")
    assert response.status_code == 200
    payload = response.json()
    assert [item["label"] for item in payload] == ["Alpha root", "Beta root"]
    assert payload[0]["path"] == str(root_a.resolve())


def test_list_endpoint_lists_directory_contents(file_browser_app: Tuple[TestClient, Path, Path]) -> None:
    client, root_a, nested = file_browser_app
    response = client.get("/api/files/list", params={"path": str(root_a)})
    assert response.status_code == 200
    payload = response.json()
    names = [entry["name"] for entry in payload["entries"]]
    assert "example.txt" in names
    assert "nested" in names
    assert payload["root"]["label"] == "Alpha root"
    assert payload["parent"] is None

    nested_response = client.get("/api/files/list", params={"path": str(nested)})
    assert nested_response.status_code == 200
    nested_payload = nested_response.json()
    assert nested_payload["parent"] == str(root_a)


def test_list_rejects_outside_allowlist(file_browser_app: Tuple[TestClient, Path, Path]) -> None:
    client, root_a, _nested = file_browser_app
    outside = root_a.parent.parent
    response = client.get("/api/files/list", params={"path": str(outside)})
    assert response.status_code == 403


def test_validate_endpoint_reports_state(file_browser_app: Tuple[TestClient, Path, Path]) -> None:
    client, root_a, nested = file_browser_app
    file_response = client.get("/api/files/validate", params={"path": str(root_a / "example.txt")})
    assert file_response.status_code == 200
    data = file_response.json()
    assert data["exists"] is True
    assert data["isFile"] is True
    assert data["isDir"] is False
    assert data["allowed"] is True

    missing_response = client.get("/api/files/validate", params={"path": str(nested / "missing.txt")})
    assert missing_response.status_code == 200
    missing = missing_response.json()
    assert missing["exists"] is False
    assert missing["allowed"] is True

    outside = root_a.parent.parent / "other"
    outside_response = client.get("/api/files/validate", params={"path": str(outside)})
    assert outside_response.status_code == 200
    outside_data = outside_response.json()
    assert outside_data["allowed"] is False


def test_validate_nonexistent_directory(file_browser_app: Tuple[TestClient, Path, Path]) -> None:
    """Verify that a non-existent path inside an allowlist root is considered allowed."""
    client, root_a, _nested = file_browser_app
    non_existent_dir = root_a / "new_folder"
    response = client.get("/api/files/validate", params={"path": str(non_existent_dir)})
    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is False
    assert data["allowed"] is True
