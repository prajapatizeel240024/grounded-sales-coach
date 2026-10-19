import pytest
from fastapi.testclient import TestClient

from fieldcoach.api import app, get_repository
from fieldcoach.store import Repository


@pytest.fixture(scope="module")
def client(settings):
    repo = Repository(settings)  # settings has no DATABASE_URL, so this serves the tracked files
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/healthz").json()
    assert body["ok"] and body["synthetic"] and body["storage"] == "files" and body["run_id"]


def test_dashboard_renders_with_synthetic_notice(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Synthetic data." in page.text
    assert "What top reps do differently" in page.text and "How far to trust this" in page.text


def test_reps_endpoint(client):
    reps = client.get("/api/reps").json()
    assert len(reps) == 5
    assert reps[0]["status"] == "needs_help"


def test_meeting_endpoint(client):
    body = client.get("/api/meetings/T01").json()
    assert body["synthetic"] is True
    assert body["transcript"]["id"] == "T01" and body["coaching"]["note"]["suggestions"]
    assert client.get("/api/meetings/T99").status_code == 404


def test_evals_endpoint(client):
    assert "golden" in client.get("/api/evals").json()["metrics"]


def test_analyze_endpoint_uses_the_offline_baseline(client, settings):
    text = (settings.transcripts_dir / "T13.txt").read_text(encoding="utf-8")
    body = client.post("/api/analyze", json={"transcript": text}).json()
    assert body["backend"] == "heuristic" and body["synthetic"] is True
    assert 0 <= body["analysis"]["score"] <= 100


def test_analyze_rejects_transcripts_not_marked_synthetic(client):
    resp = client.post("/api/analyze", json={"transcript": "id: X\n---\nREP: Hi.\n"})
    assert resp.status_code == 422
