#!/usr/bin/env python3
"""
validate_env.py — pre-flight environment check for DocuLingua.

Run this before starting the app to catch missing or invalid configuration:
    python validate_env.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

import os
import sys
import urllib.parse
from pathlib import Path

# Load .env if python-dotenv is available (optional — Docker injects vars directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fine — variables may already be in the environment


# ------------------------------------------------------------------------------
# Check definitions
# Each entry: (VAR_NAME, required, description, validator_fn | None)
# ------------------------------------------------------------------------------

def _is_postgres_url(value: str) -> tuple[bool, str]:
    if not value.startswith("postgresql://") and not value.startswith("postgres://"):
        return False, "must start with postgresql:// (SQLite is not supported)"
    try:
        parsed = urllib.parse.urlparse(value)
        if not parsed.hostname:
            return False, "missing hostname"
        if not parsed.path.lstrip("/"):
            return False, "missing database name"
    except Exception as exc:
        return False, f"parse error: {exc}"
    return True, ""


def _is_nonempty(value: str) -> tuple[bool, str]:
    if not value.strip():
        return False, "must not be empty"
    return True, ""


def _is_secret_key(value: str) -> tuple[bool, str]:
    if len(value) < 32:
        return False, f"too short ({len(value)} chars) — generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
    if value in ("replace_with_generated_secret_key", "change_me"):
        return False, "still using placeholder value — generate a real key"
    return True, ""


def _is_log_level(value: str) -> tuple[bool, str]:
    valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if value.upper() not in valid:
        return False, f"must be one of {sorted(valid)}"
    return True, ""


def _is_api_url(value: str) -> tuple[bool, str]:
    if not (value.startswith("http://") or value.startswith("https://")):
        return False, "must start with http:// or https://"
    return True, ""


CHECKS = [
    # (env var,              required, description,                        validator)
    ("DATABASE_URL",         True,  "PostgreSQL connection string",        _is_postgres_url),
    ("SECRET_KEY",           True,  "Secret key for signing tokens",       _is_secret_key),
    ("POSTGRES_USER",        True,  "PostgreSQL username",                 _is_nonempty),
    ("POSTGRES_PASSWORD",    True,  "PostgreSQL password",                 _is_nonempty),
    ("POSTGRES_DB",          True,  "PostgreSQL database name",            _is_nonempty),
    ("API_HOST",             False, "API bind host (default: 0.0.0.0)",    None),
    ("API_PORT",             False, "API port (default: 8000)",            None),
    ("API_URL",              False, "Frontend → API URL",                  _is_api_url),
    ("MODEL_DIR",            False, "Directory for trained models",        None),
    ("UPLOAD_DIR",           False, "Directory for uploaded files",        None),
    ("TRANSFORMERS_CACHE",   False, "HuggingFace model cache directory",   None),
    ("LOG_LEVEL",            False, "Logging level",                       _is_log_level),
]

# ------------------------------------------------------------------------------
# Directory checks — these paths should exist (or be creatable)
# ------------------------------------------------------------------------------

DIR_CHECKS = [
    ("MODEL_DIR",  "models"),
    ("UPLOAD_DIR", "uploads"),
]


def _banner(text: str, char: str = "─", width: int = 64) -> str:
    pad = max(0, width - len(text) - 2)
    return f"{'─' * (pad // 2)} {text} {'─' * (pad - pad // 2)}"


def run_checks() -> bool:
    failures: list[str] = []
    warnings: list[str] = []

    print()
    print("╔" + "═" * 62 + "╗")
    print("║        DocuLingua — Environment Validation                  ║")
    print("╚" + "═" * 62 + "╝")
    print()

    # --- Variable checks ---
    print(_banner("Environment Variables"))
    for var, required, description, validator in CHECKS:
        value = os.getenv(var)

        if value is None:
            if required:
                print(f"  ✗  {var:<26}  MISSING  ({description})")
                failures.append(f"{var} is not set")
            else:
                print(f"  –  {var:<26}  not set  ({description})")
            continue

        if validator:
            ok, reason = validator(value)
            if not ok:
                print(f"  ✗  {var:<26}  INVALID  — {reason}")
                if required:
                    failures.append(f"{var}: {reason}")
                else:
                    warnings.append(f"{var}: {reason}")
                continue

        # Mask secrets in output
        display = "***" if "PASSWORD" in var or "SECRET" in var or "KEY" in var else value
        print(f"  ✓  {var:<26}  {display}")

    # --- Directory checks ---
    print()
    print(_banner("Required Directories"))
    for env_var, default in DIR_CHECKS:
        path = Path(os.getenv(env_var, default))
        if path.exists():
            print(f"  ✓  {str(path):<40}  exists")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓  {str(path):<40}  created")
            except OSError as exc:
                print(f"  ✗  {str(path):<40}  CANNOT CREATE — {exc}")
                failures.append(f"Cannot create directory {path}: {exc}")

    # --- .env file check ---
    print()
    print(_banner("Configuration Files"))
    env_file = Path(".env")
    if env_file.exists():
        print(f"  ✓  .env file found")
    else:
        print(f"  ✗  .env file not found")
        print(f"       Run: cp .env.example .env  then fill in your values")
        failures.append(".env file is missing")

    example_file = Path(".env.example")
    if example_file.exists():
        print(f"  ✓  .env.example file found")
    else:
        print(f"  –  .env.example file not found")

    # --- Summary ---
    print()
    print("─" * 64)
    if failures:
        print(f"  RESULT: FAILED — {len(failures)} error(s), {len(warnings)} warning(s)\n")
        for f in failures:
            print(f"    ✗  {f}")
        if warnings:
            print()
            for w in warnings:
                print(f"    ⚠  {w}")
        print()
        print("  Fix the errors above, then re-run: python validate_env.py")
        print("─" * 64)
        print()
        return False
    elif warnings:
        print(f"  RESULT: PASSED with {len(warnings)} warning(s)\n")
        for w in warnings:
            print(f"    ⚠  {w}")
        print("─" * 64)
        print()
        return True
    else:
        print("  RESULT: ALL CHECKS PASSED — ready to start DocuLingua")
        print("─" * 64)
        print()
        return True


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
