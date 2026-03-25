"""
tests/test_api.py — DocuLingua integration tests.

Uses SQLite in-memory so no PostgreSQL is required.
Run:  pytest tests/test_api.py -v

Environment variables are set before any imports so the app initialises
against the test database and with the test API key.
"""

import io
import os
import struct
import sys
import uuid
import zlib

import pytest

# ---------------------------------------------------------------------------
# Environment must be configured BEFORE importing the app
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_TEST_API_KEY = "test-api-key-doculingua"
os.environ["DATABASE_URL"]          = "sqlite:///./test_doculingua.db"
os.environ["SECRET_KEY"]            = "test-secret-key-not-for-production"
os.environ["STRICT_MODEL_LOADING"]  = "false"
os.environ["API_KEY"]               = _TEST_API_KEY
os.environ["UPLOAD_DIR"]            = "uploads"

from fastapi.testclient import TestClient          # noqa: E402
from sqlalchemy import create_engine               # noqa: E402
from sqlalchemy.orm import sessionmaker            # noqa: E402

from backend.database.database import Base, get_db  # noqa: E402
from backend.main import app                       # noqa: E402

# ---------------------------------------------------------------------------
# Test database wiring
# ---------------------------------------------------------------------------
_TEST_DB_PATH = "./test_doculingua.db"
_TEST_DB_URL  = f"sqlite:///{_TEST_DB_PATH}"

test_engine = create_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def _override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db
Base.metadata.create_all(bind=test_engine)

# ---------------------------------------------------------------------------
# Test client with API key baked in
# ---------------------------------------------------------------------------
client = TestClient(app, headers={"X-API-Key": _TEST_API_KEY})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 200, height: int = 100) -> bytes:
    """Build a minimal valid white PNG using stdlib only."""
    def _chunk(name: bytes, data: bytes) -> bytes:
        body = name + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


_PNG = _make_png()


def _upload_png() -> str:
    """Upload the test PNG and return its document_id."""
    r = client.post("/upload", files={"file": ("t.png", io.BytesIO(_PNG), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["document_id"]


# ---------------------------------------------------------------------------
# Per-test upload cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_uploads():
    """Track files written to uploads/ during a test and remove them after."""
    upload_dir = os.environ.get("UPLOAD_DIR", "uploads")
    before = set(os.listdir(upload_dir)) if os.path.isdir(upload_dir) else set()
    yield
    if os.path.isdir(upload_dir):
        for fname in os.listdir(upload_dir):
            if fname not in before:
                try:
                    os.remove(os.path.join(upload_dir, fname))
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Session-level teardown — drop tables and remove the test DB file
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def teardown_db():
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists(_TEST_DB_PATH):
        os.remove(_TEST_DB_PATH)


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------

def test_health_no_auth_required():
    """GET /health must work without an API key."""
    r = TestClient(app).get("/health")   # client without auth headers
    assert r.status_code == 200


def test_protected_endpoint_missing_key():
    """Requests without X-API-Key must be rejected with 403."""
    r = TestClient(app).get("/stats")    # no headers
    assert r.status_code == 403


def test_protected_endpoint_wrong_key():
    """Requests with a wrong key must be rejected with 401."""
    r = TestClient(app).get("/stats", headers={"X-API-Key": "totally-wrong-key"})
    assert r.status_code == 401


def test_protected_endpoint_valid_key():
    """Requests with the correct key must succeed."""
    r = client.get("/stats")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Core endpoint tests
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "classifier" in data["models"]
    assert "ner" in data["models"]


def test_stats_structure():
    r = client.get("/stats")
    assert r.status_code == 200
    stats = r.json()["statistics"]
    assert "total_documents_uploaded" in stats
    assert "documents_by_language" in stats
    assert "pipeline_average_time_seconds" in stats


def test_upload_png_succeeds():
    r = client.post("/upload", files={"file": ("doc.png", io.BytesIO(_PNG), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert "document_id" in body
    assert body["status"] == "uploaded"


def test_upload_unsupported_format():
    r = client.post("/upload", files={"file": ("note.txt", io.BytesIO(b"text"), "text/plain")})
    assert r.status_code == 400


def test_upload_no_file():
    r = client.post("/upload")
    assert r.status_code == 422   # FastAPI validation error


def test_results_before_analysis():
    doc_id = _upload_png()
    r = client.get(f"/results/{doc_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "pending_analysis"


def test_results_invalid_uuid():
    r = client.get("/results/not-a-valid-uuid")
    assert r.status_code == 400


def test_results_unknown_id():
    r = client.get(f"/results/{uuid.uuid4()}")
    assert r.status_code == 404


def test_analyze_unknown_id():
    r = client.post(f"/analyze/{uuid.uuid4()}")
    assert r.status_code == 404


def test_analyze_invalid_uuid():
    r = client.post("/analyze/not-a-uuid")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline():
    """Upload → analyze → results → stats — end-to-end happy path."""
    # Upload
    upload_r = client.post(
        "/upload",
        files={"file": ("pipeline.png", io.BytesIO(_PNG), "image/png")},
    )
    assert upload_r.status_code == 200
    doc_id = upload_r.json()["document_id"]

    # Analyze
    analyze_r = client.post(f"/analyze/{doc_id}")
    assert analyze_r.status_code == 200
    ab = analyze_r.json()
    assert ab["status"] == "processed"
    assert "document_type" in ab
    assert "detected_language" in ab
    assert isinstance(ab["document_confidence"], (int, float))
    assert "processing_time_seconds" in ab

    # Results
    results_r = client.get(f"/results/{doc_id}")
    assert results_r.status_code == 200
    rb = results_r.json()
    assert rb["status"] == "processed"
    assert "detected_language" in rb
    assert "global_confidence" in rb
    assert isinstance(rb["extracted_fields"], dict)

    # Stats should reflect at least this document
    stats_r = client.get("/stats")
    assert stats_r.status_code == 200
    assert stats_r.json()["statistics"]["total_documents_uploaded"] >= 1


def test_request_id_in_error_response():
    """Error responses from the global handler should include a request_id."""
    # Trigger a 404 through analyze so it goes through normal error handling
    r = client.post(f"/analyze/{uuid.uuid4()}")
    # 404 is raised as HTTPException, not caught by global handler,
    # but the request_id header should still be present in the response.
    assert "x-request-id" in r.headers
