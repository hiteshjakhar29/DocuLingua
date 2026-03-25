```
██████╗  ██████╗  ██████╗██╗   ██╗██╗     ██╗███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
██╔══██╗██╔═══██╗██╔════╝██║   ██║██║     ██║████╗  ██║██╔════╝ ██║   ██║██╔══██╗
██║  ██║██║   ██║██║     ██║   ██║██║     ██║██╔██╗ ██║██║  ███╗██║   ██║███████║
██║  ██║██║   ██║██║     ██║   ██║██║     ██║██║╚██╗██║██║   ██║██║   ██║██╔══██║
██████╔╝╚██████╔╝╚██████╗╚██████╔╝███████╗██║██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
```

> **Multilingual document intelligence — extract structured data from credentials in 20+ languages.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Features

🌍 **20+ Languages** — English, Arabic, Spanish, French, Hindi, and many more, handled natively by a multilingual BERT backbone

🤖 **Three ML Models** — Dual-engine OCR (Tesseract + EasyOCR), multilingual BERT classifier, and spaCy NER pipeline working in concert

⚡ **Fast Processing** — Average end-to-end analysis in under 3 seconds per document

📊 **100% Classification Accuracy** — Across all 6 document types on the held-out evaluation set (151 documents)

🔍 **96.22% NER F1-Score** — Entity-level precision 94.7%, recall 97.8% across PERSON, DEGREE, DATE, and INSTITUTION

🎯 **Confidence-Based Flagging** — Automatic manual review routing when overall confidence < 75% or any field confidence < 60%

---

## Quick Start

```bash
git clone https://github.com/yourusername/DocuLingua
cd DocuLingua
cp .env.example .env
# Edit .env — set DATABASE_URL and API_KEY at minimum
./quick_setup.sh          # 8-step setup: DB init, model training, validation
docker-compose up
```

| Service        | URL                          |
|----------------|------------------------------|
| API (Swagger)  | http://localhost:8000/docs   |
| Streamlit UI   | http://localhost:8501        |
| PostgreSQL     | localhost:5432               |

> **Skip model training** (uses fallback models): `./quick_setup.sh --skip-train`

---

## Architecture

```
┌─────────────┐
│   Document  │  PDF / image upload
│   (PDF/IMG) │
└──────┬──────┘
       │  POST /upload
       ▼
┌─────────────────────────────────────────────────────────────┐
│                     ML PIPELINE                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  OCR ENGINE  (concurrent via ThreadPoolExecutor)      │  │
│  │  ┌─────────────────┐   ┌──────────────────────────┐  │  │
│  │  │    Tesseract     │   │        EasyOCR           │  │  │
│  │  │  (classical OCR) │   │  (deep-learning OCR)     │  │  │
│  │  └────────┬────────┘   └────────────┬─────────────┘  │  │
│  │           └──────────┬──────────────┘                │  │
│  │               Cross-engine agreement score            │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LANGUAGE DETECTION  (langdetect)                     │  │
│  │  Detects language from OCR text → routes to          │  │
│  │  multilingual model; falls back to English           │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DOCUMENT CLASSIFIER  (multilingual BERT)             │  │
│  │  6 classes: university_degree · transcript ·         │  │
│  │  professional_license · employment_letter ·          │  │
│  │  diploma · certificate                               │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  NAMED ENTITY RECOGNITION  (spaCy)                   │  │
│  │  Entities: PERSON · INSTITUTION · DEGREE ·           │  │
│  │  DATE · CERT_NUMBER · GRADE                          │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CONFIDENCE SCORING  (5-factor weighted)              │  │
│  │  ocr_agreement 30% · ocr_char_confidence 25%         │  │
│  │  ner_probability 25% · pattern_validation 10%        │  │
│  │  image_quality 10%                                   │  │
│  └──────────────────────┬───────────────────────────────┘  │
└───────────────────────────────────────────────────────────  │
                          │                                   │
                          ▼
              ┌───────────────────────┐
              │   Structured Results  │  GET /results/{id}
              │   + Review Flag       │
              └───────────────────────┘
```

### Tech Stack

| Layer          | Technology                                          |
|----------------|-----------------------------------------------------|
| API            | FastAPI 0.104, Uvicorn, Pydantic                    |
| OCR            | Tesseract 5, EasyOCR, OpenCV (deskew/denoise)       |
| Classification | `bert-base-multilingual-cased` (HuggingFace)        |
| NER            | spaCy 3.x with custom trained pipeline              |
| Database       | PostgreSQL, SQLAlchemy 2.0 ORM (UUID PKs)           |
| Frontend       | Streamlit                                           |
| Infra          | Docker Compose, HuggingFace model cache volume      |

---

## Performance Metrics

### Document Classification (BERT)

| Metric    | Target | Achieved   |
|-----------|--------|------------|
| Accuracy  | 89%    | **100%**   |
| Precision | —      | **100%**   |
| Recall    | —      | **100%**   |
| F1-score  | —      | **100%**   |

Results across 151 held-out documents, 6 document classes:

| Document Type         | Precision | Recall | F1   | Support |
|-----------------------|-----------|--------|------|---------|
| certificate           | 1.00      | 1.00   | 1.00 | 25      |
| diploma               | 1.00      | 1.00   | 1.00 | 25      |
| employment_letter     | 1.00      | 1.00   | 1.00 | 25      |
| professional_license  | 1.00      | 1.00   | 1.00 | 26      |
| transcript            | 1.00      | 1.00   | 1.00 | 25      |
| university_degree     | 1.00      | 1.00   | 1.00 | 25      |

### Named Entity Recognition (spaCy)

| Metric        | Target | Achieved     |
|---------------|--------|--------------|
| Overall F1    | 87%    | **96.22%**   |
| Precision     | —      | **94.68%**   |
| Recall        | —      | **97.80%**   |

Per-entity breakdown:

| Entity      | Precision | Recall | F1       |
|-------------|-----------|--------|----------|
| DATE        | 1.000     | 1.000  | **1.000** |
| DEGREE      | 1.000     | 1.000  | **1.000** |
| PERSON      | 0.933     | 1.000  | **0.966** |
| INSTITUTION | 0.727     | 0.800  | **0.762** |

### Processing Speed

| Stage                  | Typical Time |
|------------------------|--------------|
| OCR (dual-engine)      | 1.5–2.5 s    |
| Classification + NER   | 0.3–0.8 s    |
| **End-to-end total**   | **< 3 s**    |

---

## Project Structure

```
DocuLingua/
├── backend/
│   ├── main.py               # FastAPI app, endpoints, pipeline orchestration
│   ├── auth.py               # X-API-Key constant-time validation
│   ├── database/             # SQLAlchemy models (Document → Extraction → Entity)
│   ├── ml_models/
│   │   ├── ocr.py            # Dual-engine OCR with CV preprocessing
│   │   ├── train_classifier.py  # Multilingual BERT fine-tuning
│   │   ├── train_ner.py      # spaCy NER pipeline training
│   │   └── confidence_scorer.py # 5-factor weighted confidence
│   ├── evaluate_classifier.py
│   ├── evaluate_ner.py
│   └── evaluate_ab_test.py
├── frontend/                 # Streamlit dashboard (active)
├── models/
│   ├── classifier/           # Fine-tuned BERT artifacts (~2 GB)
│   └── ner/                  # Trained spaCy model
├── data/
│   └── synthesize.py         # Synthetic multilingual PDF generator
├── results/
│   ├── classifier_evaluation.json
│   ├── ner_evaluation.json
│   ├── confusion_matrix.png
│   └── examples_report.txt
├── tests/
│   └── test_api.py
├── test_full_pipeline.py
├── validate_env.py
├── quick_setup.sh
├── docker-compose.yml
└── .env.example
```

---

## API Documentation

All endpoints (except `GET /health`) require the header:
```
X-API-Key: <your API key>
```

### Upload a document
```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: your_api_key" \
  -F "file=@/path/to/certificate.pdf"
# Returns: { "document_id": "uuid-here" }
```

### Trigger analysis
```bash
curl -X POST http://localhost:8000/analyze/uuid-here \
  -H "X-API-Key: your_api_key"
# Returns: { "status": "processing" }
```

### Retrieve results
```bash
curl http://localhost:8000/results/uuid-here \
  -H "X-API-Key: your_api_key"
```

Example response:
```json
{
  "document_id": "uuid-here",
  "document_type": "university_degree",
  "language": "en",
  "confidence_score": 0.91,
  "requires_manual_review": false,
  "entities": [
    { "type": "PERSON",      "value": "Jane Smith",           "confidence": 0.97 },
    { "type": "INSTITUTION", "value": "MIT",                  "confidence": 0.89 },
    { "type": "DEGREE",      "value": "Bachelor of Science",  "confidence": 0.95 },
    { "type": "DATE",        "value": "2024-05-15",           "confidence": 1.00 }
  ]
}
```

Full interactive docs available at **http://localhost:8000/docs** when running.

---

## Deployment

### Local (Development)
```bash
cp .env.example .env          # Configure DATABASE_URL and API_KEY
./quick_setup.sh              # Full setup with model training
docker-compose up --build     # Start all three services
```

### Environment Variables
| Variable            | Required | Description                                   |
|---------------------|----------|-----------------------------------------------|
| `DATABASE_URL`      | Yes      | PostgreSQL connection string                  |
| `API_KEY`           | Yes      | Secret for X-API-Key header                   |
| `MODEL_DIR`         | No       | Path to trained models (default: `models/`)   |
| `UPLOAD_DIR`        | No       | Temp file cache (default: `uploads/`)         |
| `TRANSFORMERS_CACHE`| No       | HuggingFace model cache path                  |

> **Note:** Production deployment benefits from GPU acceleration for optimal BERT inference speed. Total model footprint is approximately 2 GB (BERT + spaCy + EasyOCR weights).

---

## Development

### Train Models
```bash
# Generate synthetic multilingual training PDFs
python data/synthesize.py

# Fine-tune multilingual BERT classifier
python backend/ml_models/train_classifier.py

# Train spaCy NER pipeline
python backend/ml_models/train_ner.py
```

### Evaluate Models
```bash
python backend/evaluate_classifier.py   # Confusion matrix + per-class metrics
python backend/evaluate_ner.py          # Token-level F1 / precision / recall
python backend/evaluate_ab_test.py      # Benchmark vs baseline
```

### Run Tests
```bash
pytest tests/test_api.py                                           # Unit & API tests
python test_full_pipeline.py                                       # End-to-end smoke test
python test_full_pipeline.py --api http://localhost:8000 --timeout 60
```

### Adding a New Document Type
1. Add the new label to the classifier training data in `data/synthesize.py`
2. Retrain: `python backend/ml_models/train_classifier.py`
3. Add any type-specific NER patterns to `backend/ml_models/train_ner.py` and retrain
4. Update the label enum in `backend/main.py`
5. Re-run evaluation scripts to validate metrics

---

## Future Improvements

- **More document types** — driving licenses, immigration documents, bank statements
- **Improved INSTITUTION NER** — current F1 is 76.2%; targeted training data and better boundary detection will close the gap
- **Batch processing** — `POST /analyze/batch` endpoint with async job queue
- **Cloud GPU deployment** — AWS / GCP setup guide with GPU-backed inference for production throughput
- **Real-world document hardening** — augment synthetic training data with scanned real documents to improve robustness on low-quality scans
- **Streaming results** — Server-Sent Events for long-running OCR jobs

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Contact

Built by **Hitesh**

- GitHub Issues: [github.com/yourusername/DocuLingua/issues](https://github.com/yourusername/DocuLingua/issues)

---

*Tested on Python 3.11 · macOS · Docker 24+*
