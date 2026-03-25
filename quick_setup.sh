#!/usr/bin/env bash
# =============================================================================
# quick_setup.sh — DocuLingua one-shot setup
#
# Usage:
#   chmod +x quick_setup.sh
#   ./quick_setup.sh              # full setup
#   ./quick_setup.sh --skip-train # skip model training (use fallbacks)
#
# What it does (idempotent — safe to re-run):
#   1. Validates the .env file
#   2. Creates required directories
#   3. Installs Python dependencies
#   4. Downloads required spaCy model
#   5. Generates synthetic training data  (skipped if data already exists)
#   6. Trains the document classifier    (skipped if model already exists)
#   7. Trains the NER model              (skipped if model already exists)
#   8. Initialises the PostgreSQL schema
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}  ✓${RESET}  $*"; }
warn() { echo -e "${YELLOW}  ⚠${RESET}  $*"; }
info() { echo -e "${CYAN}  →${RESET}  $*"; }
fail() { echo -e "${RED}  ✗  FATAL: $*${RESET}" >&2; exit 1; }

header() {
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}  $*${RESET}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"
}

# ── Args ─────────────────────────────────────────────────────────────────────
SKIP_TRAIN=false
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

# ── Locate project root (directory containing this script) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Virtual environment bootstrap ────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "No venv found — creating one with python3.11 …"
    python3.11 -m venv "$VENV_DIR" || fail "Failed to create venv. Is python3.11 installed?"
    ok "Created venv at $VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
ok "Activated venv: $PYTHON"

# =============================================================================
# STEP 1 — Environment
# =============================================================================
header "Step 1 / 8 — Environment"

if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        warn ".env not found — copying from .env.example"
        warn "Edit .env and set POSTGRES_PASSWORD and SECRET_KEY before running Docker."
        cp .env.example .env
    else
        fail ".env and .env.example are both missing. Cannot continue."
    fi
fi

# Load .env so we can read DATABASE_URL etc. in this script
set -a
# shellcheck source=/dev/null
source .env
set +a

if [[ -z "${DATABASE_URL:-}" ]]; then
    fail "DATABASE_URL is not set in .env. Edit .env and re-run."
fi

ok "Environment loaded"

# =============================================================================
# STEP 2 — Directories
# =============================================================================
header "Step 2 / 8 — Directories"

for dir in uploads models/classifier models/ner results data/synthetic_docs; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        ok "Created $dir/"
    else
        ok "$dir/ already exists"
    fi
done

# =============================================================================
# STEP 3 — Python dependencies
# =============================================================================
header "Step 3 / 8 — Python dependencies"

PY_VERSION=$("$PYTHON" --version 2>&1)
info "Using $PY_VERSION at $PYTHON"

REQ_FILE="backend/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
    REQ_FILE="requirements.txt"
fi
info "Installing from $REQ_FILE …"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r "$REQ_FILE"
ok "Dependencies installed"

# =============================================================================
# STEP 4 — spaCy model
# =============================================================================
header "Step 4 / 8 — spaCy language model"

SPACY_MODEL="en_core_web_trf"
if "$PYTHON" -c "import spacy; spacy.load('$SPACY_MODEL')" &>/dev/null; then
    ok "$SPACY_MODEL already downloaded"
else
    info "Downloading $SPACY_MODEL …"
    "$PYTHON" -m spacy download "$SPACY_MODEL"
    ok "$SPACY_MODEL downloaded"
fi

# =============================================================================
# STEP 5 — Synthetic training data
# =============================================================================
header "Step 5 / 8 — Synthetic training data"

METADATA="data/synthetic_docs/metadata.json"
if [[ -f "$METADATA" ]]; then
    DOC_COUNT=$("$PYTHON" -c "import json; d=json.load(open('$METADATA')); print(len(d))" 2>/dev/null || echo "?")
    ok "Synthetic data already exists ($DOC_COUNT documents) — skipping generation"
else
    info "Generating synthetic documents (this takes ~2 min) …"
    "$PYTHON" data/synthesize.py
    ok "Synthetic data generated"
fi

# =============================================================================
# STEP 6 — Train document classifier
# =============================================================================
header "Step 6 / 8 — Document classifier (multilingual BERT)"

CLASSIFIER_PATH="models/classifier"
if [[ "$SKIP_TRAIN" == "true" ]]; then
    warn "--skip-train set: skipping classifier training (zero-shot fallback will be used)"
elif [[ -f "$CLASSIFIER_PATH/config.json" ]]; then
    ok "Classifier model already exists at $CLASSIFIER_PATH — skipping training"
    info "Delete $CLASSIFIER_PATH/ and re-run to retrain."
else
    info "Training classifier — this downloads ~1 GB of weights and may take 10–30 min …"
    "$PYTHON" backend/ml_models/train_classifier.py
    if [[ -f "$CLASSIFIER_PATH/config.json" ]]; then
        ok "Classifier trained and saved to $CLASSIFIER_PATH/"
    else
        fail "Classifier training finished but model not found at $CLASSIFIER_PATH/config.json"
    fi
fi

# =============================================================================
# STEP 7 — Train NER model
# =============================================================================
header "Step 7 / 8 — NER model (spaCy fine-tune)"

NER_PATH="models/ner"
OCR_CACHE="data/synthetic_docs/ocr_cache.json"

if [[ "$SKIP_TRAIN" == "true" ]]; then
    warn "--skip-train set: skipping NER training (spaCy fallback will be used)"
elif [[ -f "$NER_PATH/meta.json" ]]; then
    ok "NER model already exists at $NER_PATH — skipping training"
    info "Delete $NER_PATH/ and re-run to retrain."
else
    if [[ ! -f "$OCR_CACHE" ]]; then
        fail "OCR cache not found at $OCR_CACHE. The classifier training step generates this file. Re-run without --skip-train."
    fi
    info "Training NER model — may take 5–15 min …"
    "$PYTHON" backend/ml_models/train_ner.py
    if [[ -f "$NER_PATH/meta.json" ]]; then
        ok "NER model trained and saved to $NER_PATH/"
    else
        fail "NER training finished but model not found at $NER_PATH/meta.json"
    fi
fi

# =============================================================================
# STEP 8 — Database initialisation
# =============================================================================
header "Step 8 / 8 — Database schema"

info "Running init_db.py (creates tables if they don't exist) …"
if "$PYTHON" backend/database/init_db.py; then
    ok "Database schema ready"
else
    warn "init_db.py failed — is PostgreSQL running and DATABASE_URL correct?"
    warn "If using Docker, start the db service first:"
    warn "  docker compose up -d db"
    warn "Then re-run: ./quick_setup.sh"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Setup complete!${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  Start with Docker:    ${CYAN}docker compose up${RESET}"
echo -e "  Start API locally:    ${CYAN}uvicorn backend.main:app --reload${RESET}"
echo -e "  Start frontend:       ${CYAN}streamlit run frontend/app.py${RESET}"
echo -e "  Validate env:         ${CYAN}python validate_env.py${RESET}"
echo -e "  Test the pipeline:    ${CYAN}python test_full_pipeline.py${RESET}"
echo ""
