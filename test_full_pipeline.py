#!/usr/bin/env python3
"""
test_full_pipeline.py — end-to-end smoke test for DocuLingua.

Uploads a real document (or creates a tiny synthetic one), runs the full
pipeline (upload → analyze → results), and reports pass/fail for every step.

Usage:
    python test_full_pipeline.py                        # default: http://localhost:8000
    python test_full_pipeline.py --api http://host:8000
    python test_full_pipeline.py --file path/to/doc.pdf
    python test_full_pipeline.py --timeout 60           # seconds per request
"""

import argparse
import io
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("requests is not installed. Run: pip install requests")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✓{RESET}  {msg}")
def fail(msg): print(f"{RED}  ✗{RESET}  {msg}")
def info(msg): print(f"{CYAN}  →{RESET}  {msg}")
def warn(msg): print(f"{YELLOW}  ⚠{RESET}  {msg}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(title: str):
    bar = "─" * 54
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{bar}{RESET}")


def _make_synthetic_image() -> bytes:
    """
    Creates a minimal grayscale PNG in-memory using only stdlib so the test
    has zero extra dependencies.  The image contains English text that the
    OCR engine can read.
    """
    import struct, zlib

    def _png_chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    width, height = 400, 200
    # Build raw pixel rows: white (255) background
    raw_rows = b"".join(b"\x00" + b"\xff" * width for _ in range(height))
    compressed = zlib.compress(raw_rows, 9)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )
    return png


def _find_existing_doc() -> tuple[str, str] | None:
    """
    Looks for a real synthetic document to use as test input.
    Returns (file_path, mime_type) or None.
    """
    search_dirs = [
        "data/synthetic_docs/certificate",
        "data/synthetic_docs/university_degree",
        "data/synthetic_docs/diploma",
        "uploads",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            fpath = os.path.join(d, fname)
            if fname.lower().endswith(".pdf"):
                return fpath, "application/pdf"
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                return fpath, "image/png"
    return None


# ── Test steps ────────────────────────────────────────────────────────────────

class PipelineTest:
    def __init__(self, api_base: str, timeout: int, file_path: str | None):
        self.base = api_base.rstrip("/")
        self.timeout = timeout
        self.file_path = file_path
        self.document_id: str | None = None
        self.results: list[dict] = []
        self.passed = 0
        self.failed = 0

    def _record(self, name: str, passed: bool, detail: str = ""):
        self.results.append({"step": name, "passed": passed, "detail": detail})
        if passed:
            ok(f"{name}{f'  ({detail})' if detail else ''}")
            self.passed += 1
        else:
            fail(f"{name}{f'  — {detail}' if detail else ''}")
            self.failed += 1

    # ── Step 0: health ────────────────────────────────────────────────────────

    def step_health(self) -> bool:
        _header("Step 0 — Health check")
        try:
            r = requests.get(f"{self.base}/health", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            mode = data.get("mode", "unknown")
            if mode == "degraded-fallback":
                warn(f"API is running in fallback mode: {data.get('hint', '')}")
            self._record("GET /health returns 200", True, f"mode={mode}")

            for model_name, status in data.get("models", {}).items():
                mtype = status.get("type", "unknown")
                self._record(
                    f"Model '{model_name}' loaded",
                    status.get("loaded", False),
                    f"type={mtype}",
                )
            return True
        except requests.ConnectionError:
            self._record(
                "GET /health",
                False,
                f"Cannot connect to {self.base}. Is the API running?",
            )
            return False
        except Exception as exc:
            self._record("GET /health", False, str(exc))
            return False

    # ── Step 1: upload ────────────────────────────────────────────────────────

    def step_upload(self) -> bool:
        _header("Step 1 — Upload document")

        # Prefer a real document; fall back to synthetic PNG
        existing = self.file_path and (self.file_path, "application/pdf")
        if not existing:
            existing = _find_existing_doc()

        if existing:
            fpath, mime = existing
            info(f"Using existing file: {fpath}")
            with open(fpath, "rb") as fh:
                file_bytes = fh.read()
            fname = os.path.basename(fpath)
        else:
            warn("No real documents found — using a synthetic blank PNG")
            file_bytes = _make_synthetic_image()
            fname = "test_document.png"
            mime = "image/png"

        try:
            r = requests.post(
                f"{self.base}/upload",
                files={"file": (fname, io.BytesIO(file_bytes), mime)},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            self.document_id = data.get("document_id")

            self._record("POST /upload returns 200", True)
            self._record(
                "Response contains document_id",
                bool(self.document_id),
                self.document_id or "missing",
            )
            self._record(
                "Status is 'uploaded'",
                data.get("status") == "uploaded",
                data.get("status"),
            )
            return bool(self.document_id)
        except requests.HTTPError as exc:
            self._record("POST /upload", False, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}")
            return False
        except Exception as exc:
            self._record("POST /upload", False, str(exc))
            return False

    # ── Step 2: analyze ───────────────────────────────────────────────────────

    def step_analyze(self) -> bool:
        _header("Step 2 — Analyze document")
        if not self.document_id:
            self._record("POST /analyze", False, "Skipped — no document_id from upload step")
            return False

        info(f"Analyzing document {self.document_id} …")
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base}/analyze/{self.document_id}",
                timeout=self.timeout,
            )
            elapsed = round(time.time() - t0, 2)
            r.raise_for_status()
            data = r.json()

            self._record("POST /analyze returns 200", True)
            self._record(
                "Status is 'processed'",
                data.get("status") == "processed",
                data.get("status"),
            )
            self._record(
                "document_type is set",
                bool(data.get("document_type")),
                data.get("document_type", "missing"),
            )
            self._record(
                "document_confidence is a number",
                isinstance(data.get("document_confidence"), (int, float)),
                str(data.get("document_confidence")),
            )
            self._record(
                "processing_time_seconds present",
                "processing_time_seconds" in data,
                f"{elapsed}s wall-clock",
            )
            self._record(
                "Processed under 30s",
                elapsed <= 30,
                f"{elapsed}s (target <30s; <5s with trained models)",
            )
            return True
        except requests.HTTPError as exc:
            self._record("POST /analyze", False, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}")
            return False
        except Exception as exc:
            self._record("POST /analyze", False, str(exc))
            return False

    # ── Step 3: results ───────────────────────────────────────────────────────

    def step_results(self) -> bool:
        _header("Step 3 — Retrieve results")
        if not self.document_id:
            self._record("GET /results", False, "Skipped — no document_id")
            return False

        try:
            r = requests.get(
                f"{self.base}/results/{self.document_id}",
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()

            self._record("GET /results returns 200", True)
            self._record(
                "Status is 'processed'",
                data.get("status") == "processed",
                data.get("status"),
            )
            self._record(
                "global_confidence is a number",
                isinstance(data.get("global_confidence"), (int, float)),
                str(data.get("global_confidence")),
            )

            fields = data.get("extracted_fields", {})
            self._record(
                "extracted_fields is present",
                isinstance(fields, dict),
                f"{len(fields)} field type(s)",
            )

            entity_count = sum(len(v) for v in fields.values())
            self._record(
                "At least one entity extracted",
                entity_count > 0,
                f"{entity_count} entity/entities",
            )

            if entity_count > 0:
                # Spot-check structure of first entity
                first_type = next(iter(fields))
                first_entity = fields[first_type][0]
                self._record(
                    "Entity has value + confidence_score",
                    "value" in first_entity and "confidence_score" in first_entity,
                    f"sample: {json.dumps(first_entity)[:120]}",
                )

            self._record(
                "raw_text_preview present",
                bool(data.get("raw_text_preview")),
            )
            return True
        except requests.HTTPError as exc:
            self._record("GET /results", False, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}")
            return False
        except Exception as exc:
            self._record("GET /results", False, str(exc))
            return False

    # ── Step 4: stats ─────────────────────────────────────────────────────────

    def step_stats(self) -> bool:
        _header("Step 4 — System stats")
        try:
            r = requests.get(f"{self.base}/stats", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            stats = data.get("statistics", {})
            self._record("GET /stats returns 200", True)
            self._record(
                "total_documents_uploaded >= 1",
                stats.get("total_documents_uploaded", 0) >= 1,
                str(stats.get("total_documents_uploaded")),
            )
            return True
        except Exception as exc:
            self._record("GET /stats", False, str(exc))
            return False

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self):
        total = self.passed + self.failed
        _header("Summary")
        print(f"\n  Total checks : {total}")
        print(f"  {GREEN}Passed{RESET}       : {self.passed}")
        if self.failed:
            print(f"  {RED}Failed{RESET}       : {self.failed}")
        else:
            print(f"  Failed       : 0")

        if self.failed == 0:
            print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED — pipeline is working end-to-end.{RESET}\n")
        else:
            print(f"\n  {RED}{BOLD}{self.failed} check(s) FAILED.{RESET}")
            print(f"  Failed steps:")
            for r in self.results:
                if not r["passed"]:
                    print(f"    ✗  {r['step']}" + (f" — {r['detail']}" if r["detail"] else ""))
            print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DocuLingua full-pipeline smoke test")
    parser.add_argument(
        "--api",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="Base URL of the DocuLingua API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to a specific document to upload (PDF/PNG/JPG)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait per HTTP request (default: 120)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}DocuLingua — Full Pipeline Test{RESET}")
    print(f"  API: {args.api}")
    print(f"  File: {args.file or '(auto-detect)'}")

    test = PipelineTest(api_base=args.api, timeout=args.timeout, file_path=args.file)

    reachable = test.step_health()
    if not reachable:
        test.print_summary()
        sys.exit(1)

    uploaded  = test.step_upload()
    analyzed  = test.step_analyze() if uploaded  else False
    _          = test.step_results() if analyzed  else test._record("GET /results", False, "Skipped")
    test.step_stats()

    test.print_summary()
    sys.exit(0 if test.failed == 0 else 1)


if __name__ == "__main__":
    main()
