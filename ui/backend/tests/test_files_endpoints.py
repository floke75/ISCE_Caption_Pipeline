import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from ui.backend.api.routes.files import FileBrowser, create_file_router
from fastapi import FastAPI

@pytest.fixture
def tmp_roots(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test.txt").write_text("Hello World", encoding="utf-8")
    (workspace / "secret.key").write_text("SECRET", encoding="utf-8")

    # Nested dir
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "nested.json").write_text("{}", encoding="utf-8")

    return [("workspace", "Workspace", workspace)]

@pytest.fixture
def client(tmp_roots):
    browser = FileBrowser(tmp_roots)
    app = FastAPI()
    app.include_router(create_file_router(browser))
    return TestClient(app)

def test_get_content(client, tmp_roots):
    workspace = tmp_roots[0][2]
    path = str(workspace / "test.txt")

    response = client.get(f"/api/files/content?path={path}")
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Hello World"
    assert data["size"] == 11
    assert data["truncated"] is False

def test_get_content_outside_root(client, tmp_path):
    # Try to access a file outside the workspace
    outside = tmp_path / "outside.txt"
    outside.write_text("Forbidden")

    response = client.get(f"/api/files/content?path={outside}")
    assert response.status_code == 403

def test_download_file(client, tmp_roots):
    workspace = tmp_roots[0][2]
    path = str(workspace / "test.txt")

    response = client.get(f"/api/files/download?path={path}")
    assert response.status_code == 200
    assert response.content == b"Hello World"

def test_download_outside_root(client, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("Forbidden")

    response = client.get(f"/api/files/download?path={outside}")
    assert response.status_code == 403
