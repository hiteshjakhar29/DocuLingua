# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocuLingua is a multilingual document intelligence system that extracts structured data from educational and professional credentials (degrees, transcripts, licenses) across 20+ languages. It uses a dual-engine OCR pipeline (Tesseract + EasyOCR), multilingual BERT classification, spaCy NER, and a 5-factor confidence scoring system.

## Commands

### Setup
```bash
chmod +x quick_setup.sh
./quick_setup.sh              # Full 8-step setup (DB init, model training, etc.)
./quick_setup.sh --skip-train # Skip ML model training (uses fallback models)
python validate_env.py        # Validate environment variables before startup
```

### Running Services
```bash
docker-compose up --build     # Start all services (PostgreSQL, FastAPI, Streamlit)
# API Swagger UI: http://localhost:8000/docs
# Streamlit UI:   http://localhost:8501
# PostgreSQL:     localhost:5432
```

### Testing
```bash
pytest tests/test_api.py                                          # Unit/API tests
python test_full_pipeline.py                                      # End-to-end smoke test
python test_full_pipeline.py --api http://localhost:8000 --timeout 60  # With options
```

### ML Training & Evaluation
```bash
python data/synthesize.py                        # Generate synthetic training PDFs
python backend/ml_models/train_classifier.py     # Fine-tune multilingual BERT
python backend/ml_models/train_ner.py            # Train spaCy NER pipeline
python backend/evaluate_classifier.py            # Confusion matrix + per-class metrics
python backend/evaluate_ner.py                   # Token-level F1/precision/recall
python backend/evaluate_ab_test.py               # Benchmark against baseline
```

## Architecture

### Service Layout
- **`backend/`** — FastAPI app (`main.py`), auth (`auth.py`), DB layer (`database/`), ML models (`ml_models/`)
- **`frontend/`** — Streamlit dashboard (supersedes `streamlit_app/` which is legacy)
- **`models/`** — Trained model artifacts: `classifier/` (BERT) and `ner/` (spaCy)
- **`data/`** — Synthetic PDF generation scripts and generated training data

### ML Pipeline (triggered by `POST /analyze/{document_id}`)
1. **OCR** (`ml_models/ocr.py`): CV preprocessing (deskew, denoise, binarize) → concurrent Tesseract + EasyOCR via `ThreadPoolExecutor` → cross-engine agreement score
2. **Language Detection** (`main.py`): `langdetect.detect()` on OCR text, falls back to English for <20 chars or unsupported languages
3. **Classification** (`ml_models/train_classifier.py`): Multilingual BERT → 6 document types (university_degree, transcript, professional_license, employment_letter, diploma, certificate)
4. **NER** (`ml_models/train_ner.py`): spaCy pipeline → PERSON, INSTITUTION, DEGREE, DATE, CERT_NUMBER, GRADE entities
5. **Confidence Scoring** (`ml_models/confidence_scorer.py`): 5-factor weighted score — ocr_agreement (30%), ocr_char_confidence (25%), ner_probability (25%), pattern_validation (10%), image_quality (10%)

### Database (PostgreSQL required — SQLite rejected at startup)
Three tables: `Document` → `Extraction` → `Entity` (UUID PKs, FK relationships). SQLAlchemy 2.0 ORM with connection pooling and pre-ping. DB URL from `DATABASE_URL` env var.

### API Authentication
All endpoints except `GET /health` require `X-API-Key` header. Validated via constant-time comparison in `auth.py`. CORS is currently wide-open (`allow_origins=["*"]`).

### Key Design Decisions
- **Graceful model degradation**: Classifier and NER are optional; pipeline runs with reduced accuracy if models aren't trained
- **Manual review flagging**: Entities flagged when overall confidence < 75% or field confidence < 60%
- **Request correlation**: `ContextVar` injects request IDs into all log lines for async tracing
- **PDF handling**: Rasterized to PNG at analysis time (not upload), temp files cleaned up after OCR

## Configuration

Copy `.env.example` to `.env`. Critical variables:
- `DATABASE_URL` — PostgreSQL connection string (required)
- `API_KEY` — Secret for `X-API-Key` header
- `MODEL_DIR=models` — Path to trained model artifacts
- `UPLOAD_DIR=uploads` — Temporary file cache
- `TRANSFORMERS_CACHE` — HuggingFace model cache (persisted as Docker volume `hf_cache:`)

## Known Issues
- `resume_training()` in `train_ner.py` is a deprecated spaCy API — should use `update()`
- Some bare `except` blocks in the ML pipeline need more specific error types
