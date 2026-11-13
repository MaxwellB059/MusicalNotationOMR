"""S8a gate: DB layer (mongomock), stateless fallback, API contract, CORS, secret hygiene."""
import re
import subprocess
from pathlib import Path

import cv2
import mongomock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from omr import api, db as dbmod

ROOT = Path(__file__).resolve().parents[2]


def mem_db():
    return dbmod.get_db(client_factory=lambda: mongomock.MongoClient())


def png_bytes():
    ok, buf = cv2.imencode(".png", np.full((40, 80, 3), 255, np.uint8))
    return buf.tobytes()


def test_no_uri_means_stateless_not_crash(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert dbmod.get_db() is None
    assert dbmod.create_job(None, "x.png", 3)           # still hands back an id
    dbmod.update_job(None, "id", status="x")             # all helpers are no-ops on None
    dbmod.save_result(None, "id", "<xml/>")


def test_unreachable_db_falls_back_to_stateless():
    def boom():
        raise RuntimeError("cannot reach atlas")
    assert dbmod.get_db(client_factory=boom) is None


def test_job_lifecycle_and_collections():
    db = mem_db()
    jid = dbmod.create_job(db, "a.png", 10)
    assert dbmod.get_job(db, jid)["status"] == "queued"
    dbmod.update_job(db, jid, status="running", stage="staff")
    assert dbmod.get_job(db, jid)["stage"] == "staff"
    dbmod.save_result(db, jid, "<score/>", detections=[{"label": "note_quarter"}], model_versions={"cnn": "v1"})
    assert dbmod.get_job(db, jid)["status"] == "done"
    assert db.results.find_one({"job_id": jid})["musicxml"] == "<score/>"
    dbmod.save_correction(db, jid, "s1", "note_quarter", "note_eighth")
    assert db.corrections.count_documents({"job_id": jid}) == 1
    dbmod.record_eval(db, "v1", {"overall": 0.9})
    assert db.eval_runs.find_one({"model_version": "v1"})["metrics"]["overall"] == 0.9
    assert dbmod.get_job(db, "nope") is None


def test_indexes_created_including_ttl():
    db = mem_db()
    info = db.jobs.index_information()
    assert any("expireAfterSeconds" in v for v in info.values())


def test_api_backward_compatible_and_health():
    c = TestClient(api.create_app(lambda img: "<score-partwise/>", db=mem_db()))
    r = c.post("/api/omr", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and r.text == "<score-partwise/>" and "x-job-id" in r.headers
    assert c.get("/health").json() == {"status": "ok", "db": "connected"}
    assert c.get(f"/api/jobs/{r.headers['x-job-id']}").json()["status"] == "done"


def test_api_stateless_mode():
    c = TestClient(api.create_app(lambda img: "<x/>", db=None))
    assert c.get("/health").json()["db"] == "stateless"
    assert c.post("/api/omr", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 200
    assert c.get("/api/jobs/abc").status_code == 404


def test_api_errors_are_structured():
    c = TestClient(api.create_app(lambda img: "<x/>"))
    assert c.post("/api/omr", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 400

    def bad(img):
        raise ValueError("boom")
    c2 = TestClient(api.create_app(bad, db=mem_db()), raise_server_exceptions=False)
    assert c2.post("/api/omr", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 500


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_BYTES", 100)
    c = TestClient(api.create_app(lambda img: "<x/>"))
    assert c.post("/api/omr", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 413


def test_cors_allowlist_and_localhost():
    c = TestClient(api.create_app(lambda img: "<x/>", origins=["https://secundo.up.railway.app"]))
    ok = c.options("/api/omr", headers={"Origin": "https://secundo.up.railway.app",
                                        "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://secundo.up.railway.app"
    local = c.options("/api/omr", headers={"Origin": "http://localhost:8082",
                                           "Access-Control-Request-Method": "POST"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:8082"
    bad = c.options("/api/omr", headers={"Origin": "https://evil.example",
                                         "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_origins_env_parsing(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com/, https://b.com ,")
    assert api.allowed_origins_from_env() == ["https://a.com", "https://b.com"]


def test_no_mongo_credentials_committed():
    pattern = re.compile(r"mongodb(\+srv)?://[^\s'\"<>]+:[^\s'\"<>@]+@")
    for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split():
        p = ROOT / f
        if p.suffix in {".py", ".md", ".yaml", ".yml", ".json", ".env", ".txt", ".toml", ".ts", ".tsx"} and p.exists():
            if f.endswith(("test_s8a_api_db.py", "PLAN.md", "TRAINING_RUNBOOK.md")):
                continue
            assert not pattern.search(p.read_text(encoding="utf-8", errors="ignore")), f
