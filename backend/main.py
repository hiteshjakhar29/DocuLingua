import os
import time
import uuid
import shutil
import logging
import logging.handlers
import traceback
import tempfile
from contextvars import ContextVar
from typing import Dict, Any, List

import cv2
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from langdetect import detect as langdetect_detect, LangDetectException

# Database
from backend.database.database import engine, Base, get_db, verify_connection
from backend.database.models import Document, Extraction, Entity

# Authentication
from backend.auth import verify_api_key

# ML components
from backend.ml_models.ocr import OCREngine
from backend.ml_models.confidence_scorer import ConfidenceScorer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy

# ---------------------------------------------------------------------------
# Logging — console + rotating file (logs/error.log)
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] [req=%(request_id)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Root logger writes INFO+ to stderr
_console = logging.StreamHandler()
_console.setFormatter(_fmt)

# Rotating file handler captures WARNING+ (errors, warnings) to disk
_file_handler = logging.handlers.RotatingFileHandler(
    "logs/error.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(_fmt)


class _RequestIdFilter(logging.Filter):
    """Injects the current request_id into every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")
        return True


_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

logging.basicConfig(level=logging.INFO, handlers=[_console, _file_handler])
for _h in (_console, _file_handler):
    _h.addFilter(_RequestIdFilter())

logger = logging.getLogger("DocuLingua.API")

# ---------------------------------------------------------------------------
# Model state — populated by load_models(); read by /health
# ---------------------------------------------------------------------------
model_status: dict = {
    "classifier": {"loaded": False, "type": None, "path": None},
    "ner":        {"loaded": False, "type": None, "path": None},
}

# Create tables (idempotent)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DocuLingua API",
    version="2.0.0",
    description="Multilingual document intelligence — extracts structured data from certificates, transcripts, and licenses.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------
_SUPPORTED_OCR_LANGS = {"en", "ar", "es", "fr", "hi", "de", "pt", "zh", "ru", "ur"}
_LANGDETECT_NORMALISE = {"zh-cn": "zh", "zh-tw": "zh"}


def _detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        raw = langdetect_detect(text)
        code = _LANGDETECT_NORMALISE.get(raw, raw)
        if code not in _SUPPORTED_OCR_LANGS:
            logger.info(f"Detected language '{raw}' not in OCR support list; using 'en'.")
            return "en"
        logger.info(f"Detected document language: '{code}' (raw='{raw}')")
        return code
    except LangDetectException:
        logger.warning("langdetect failed (too little text?); defaulting to 'en'.")
        return "en"


# ---------------------------------------------------------------------------
# ML globals
# ---------------------------------------------------------------------------
ocr_engine = OCREngine()
scorer = ConfidenceScorer()

classifier_pipeline = None
ner_nlp = None

CLASS_NAMES = [
    "university_degree", "transcript", "professional_license",
    "employment_letter", "diploma", "certificate",
]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    _load_models()
    try:
        verify_connection(max_attempts=3)
        logger.info("Database connection verified.")
    except RuntimeError as exc:
        logger.error(f"Database unreachable at startup: {exc}")
        raise  # propagate so uvicorn aborts


def _load_models() -> None:
    global classifier_pipeline, ner_nlp, model_status
    logger.info("Initializing ML models…")
    strict = os.getenv("STRICT_MODEL_LOADING", "false").lower() == "true"

    # Classifier
    classifier_path = os.getenv("MODEL_DIR", "models") + "/classifier"
    if os.path.exists(os.path.join(classifier_path, "config.json")):
        logger.info(f"Loading custom classifier from {classifier_path}…")
        tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
        classifier_pipeline = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True
        )
        model_status["classifier"] = {"loaded": True, "type": "custom", "path": classifier_path}
    else:
        if strict:
            raise RuntimeError(
                f"Custom classifier not found at '{classifier_path}'. "
                "Run quick_setup.sh, or set STRICT_MODEL_LOADING=false to use zero-shot fallback."
            )
        logger.warning(f"No custom classifier at '{classifier_path}'. Classifier disabled — train the model first.")
        classifier_pipeline = None
        model_status["classifier"] = {"loaded": False, "type": "not-trained", "path": None}

    # NER
    ner_path = os.getenv("MODEL_DIR", "models") + "/ner"
    if os.path.exists(ner_path):
        logger.info(f"Loading custom NER from {ner_path}…")
        ner_nlp = spacy.load(ner_path)
        model_status["ner"] = {"loaded": True, "type": "custom", "path": ner_path}
    else:
        if strict:
            raise RuntimeError(
                f"Custom NER not found at '{ner_path}'. "
                "Run quick_setup.sh, or set STRICT_MODEL_LOADING=false to use spaCy fallback."
            )
        logger.warning(f"No custom NER at '{ner_path}'. Using spaCy fallback.")
        try:
            ner_nlp = spacy.load("en_core_web_trf")
            model_status["ner"] = {"loaded": True, "type": "spacy-trf-fallback", "path": None}
        except OSError:
            ner_nlp = spacy.load("en_core_web_sm")
            model_status["ner"] = {"loaded": True, "type": "spacy-sm-fallback", "path": None}

    logger.info("All ML models initialized.")


# ---------------------------------------------------------------------------
# Middleware — attach request_id so every log line is traceable
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_and_logging(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]  # short 8-char prefix is enough for correlation
    _request_id_ctx.set(req_id)
    request.state.request_id = req_id

    t0 = time.time()
    response = await call_next(request)
    elapsed = time.time() - t0

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed:.3f}s)"
    )
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler — last-resort catch for anything that slips through
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = getattr(request.state, "request_id", "-")
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc!r}\n"
        + traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again or contact support.",
            "request_id": req_id,
        },
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def _run_ner(text: str) -> List[Dict[str, Any]]:
    doc = ner_nlp(text)
    entities = []
    seen_spans: set = set()
    for ent in doc.ents:
        key = (ent.start_char, ent.end_char)
        if key not in seen_spans:
            entities.append({
                "entity_type": ent.label_,
                "entity_value": ent.text.strip(),
                "confidence": 0.95,
            })
            seen_spans.add(key)
    return entities


def _run_classification(text: str) -> str:
    if classifier_pipeline is None:
        logger.warning("Classifier not loaded — returning default class. Train the model first.")
        return CLASS_NAMES[0]
    try:
        res = classifier_pipeline(text[:2000])
        if isinstance(res, list) and isinstance(res[0], list):
            top = max(res[0], key=lambda x: x["score"])
            return top["label"]
        if isinstance(res, dict) and "labels" in res:
            scores = classifier_pipeline(text[:2000], candidate_labels=CLASS_NAMES)
            return scores["labels"][0]
        return CLASS_NAMES[0]
    except (ValueError, RuntimeError) as exc:
        logger.error(f"Classification failed: {exc}")
        return "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Public endpoint — no API key required."""
    all_custom = all(m["type"] == "custom" for m in model_status.values())
    return {
        "status": "ok",
        "mode": "production" if all_custom else "degraded-fallback",
        "models": model_status,
        "hint": (
            None if all_custom
            else "Run quick_setup.sh to train custom models for full accuracy."
        ),
    }


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Step 1: Persist the uploaded file and create a database record."""
    req_id = _request_id_ctx.get("-")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in {"jpg", "jpeg", "png", "pdf"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Accepted: jpg, jpeg, png, pdf.",
        )

    file_id = str(uuid.uuid4())
    file_path = f"{os.getenv('UPLOAD_DIR', 'uploads')}/{file_id}.{ext}"

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except (OSError, IOError, PermissionError) as exc:
        logger.error(f"[{req_id}] Could not write upload to disk: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

    try:
        doc = Document(
            id=file_id,
            file_path=file_path,
            filename=file.filename,
            file_size=file.size,
            file_type=ext,
        )
        db.add(doc)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        logger.error(f"[{req_id}] DB integrity error on upload: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=409, detail="Document record already exists.")
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"[{req_id}] DB error on upload: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database error while saving document.")

    return {"document_id": file_id, "status": "uploaded"}


@app.post("/analyze/{document_id}", dependencies=[Depends(verify_api_key)])
async def analyze_document(document_id: str, db: Session = Depends(get_db)):
    """Step 2: Run the full OCR → classify → NER → score pipeline."""
    req_id = _request_id_ctx.get("-")
    start_time = time.time()

    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format.")

    # Fetch document record
    try:
        doc = db.query(Document).filter(Document.id == str(doc_uuid)).first()
    except SQLAlchemyError as exc:
        logger.error(f"[{req_id}] DB error fetching document {document_id}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database error.")

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # ---------- OCR ----------
    target_image = doc.file_path
    tmp_img: str | None = None
    detected_lang = "en"

    try:
        # Rasterise PDF → PNG
        if target_image.endswith(".pdf"):
            pdf_doc = fitz.open(target_image)
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_img = tmp.name
                pix.save(tmp_img)
            pdf_doc.close()
            target_image = tmp_img

        # Quick English Tesseract pass → detect language
        preprocessed = ocr_engine.preprocess_image(target_image)
        quick = ocr_engine.extract_text_tesseract(preprocessed, ["en"])
        detected_lang = _detect_language(quick.get("text", ""))

        # Full dual-engine OCR in the detected language
        ocr_results = ocr_engine.run_dual_ocr(target_image, langs=[detected_lang])
        if "error" in ocr_results:
            raise ValueError(ocr_results["error"])

    except FileNotFoundError as exc:
        logger.error(f"[{req_id}] Image file not found for {document_id}: {exc}")
        raise HTTPException(status_code=404, detail="Uploaded file not found on disk.")
    except (cv2.error, ValueError, RuntimeError) as exc:
        logger.error(f"[{req_id}] OCR/image-processing failure for {document_id}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail="Image could not be processed by OCR engine.")
    except (OSError, IOError) as exc:
        logger.error(f"[{req_id}] I/O error during OCR for {document_id}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="I/O error during document processing.")
    finally:
        if tmp_img and os.path.exists(tmp_img):
            os.remove(tmp_img)

    combined_text = ocr_results.get("extracted_text", "")
    if not combined_text.strip():
        logger.warning(f"[{req_id}] Empty OCR output for {document_id}")

    # ---------- Classification + NER + Scoring ----------
    try:
        doc_class = _run_classification(combined_text)
        extracted_entities = _run_ner(combined_text)
    except (RuntimeError, ValueError) as exc:
        logger.error(f"[{req_id}] ML inference error for {document_id}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="ML model inference failed.")

    confidence_report_fields = []
    for ent in extracted_entities:
        score_meta = scorer.calculate_field_confidence(
            field_name=ent["entity_type"],
            field_value=ent["entity_value"],
            ocr_data={
                "agreement_percentage": ocr_results.get("agreement_percentage", 50.0),
                "character_confidence": ocr_results.get(
                    "character_confidence_scores", {}
                ).get("overall_confidence", 0.5),
            },
            ner_data={"probability": ent["confidence"]},
        )
        ent["holistic_confidence"] = score_meta["confidence_score"] / 100.0
        confidence_report_fields.append(score_meta)

    overall_report = scorer.generate_confidence_report(confidence_report_fields)
    global_confidence = overall_report.get("document_confidence_average", 0.0)
    manual_review = overall_report.get("requires_manual_review", True)
    process_time = time.time() - start_time

    # ---------- Persist ----------
    try:
        extraction = Extraction(
            document_id=doc.id,
            document_class=doc_class,
            detected_language=detected_lang,
            raw_text=combined_text[:2000],
            processing_time_seconds=process_time,
        )
        db.add(extraction)
        db.flush()

        for ent in extracted_entities:
            db.add(Entity(
                extraction_id=extraction.id,
                entity_type=ent["entity_type"],
                entity_value=ent["entity_value"],
                confidence_score=ent["holistic_confidence"],
            ))

        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"[{req_id}] DB error persisting extraction for {document_id}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database error while saving extraction results.")

    return {
        "status": "processed",
        "document_id": document_id,
        "document_type": doc_class,
        "detected_language": detected_lang,
        "document_confidence": global_confidence,
        "manual_review_required": manual_review,
        "flagged_fields_count": overall_report.get("flagged_fields_count", 0),
        "processing_time_seconds": round(process_time, 2),
        "ocr_metrics": {
            "agreement_percentage": ocr_results.get("agreement_percentage", 0.0),
            "extracted_length": len(combined_text),
        },
    }


@app.get("/results/{document_id}", dependencies=[Depends(verify_api_key)])
async def get_results(document_id: str, db: Session = Depends(get_db)):
    """Step 3: Return the structured extraction payload for a processed document."""
    req_id = _request_id_ctx.get("-")

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format.")

    try:
        doc = db.query(Document).filter(Document.id == str(doc_uuid)).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        extraction = db.query(Extraction).filter(Extraction.document_id == str(doc_uuid)).first()
        if not extraction:
            return {"document_id": document_id, "status": "pending_analysis"}

        entities = db.query(Entity).filter(Entity.extraction_id == extraction.id).all()
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error(f"[{req_id}] DB error fetching results for {document_id}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database error while retrieving results.")

    extracted_fields: Dict[str, list] = {}
    confidence_tally = 0.0
    flag_count = 0

    for e in entities:
        if e.entity_type not in extracted_fields:
            extracted_fields[e.entity_type] = []
        conf_percent = (e.confidence_score or 0.0) * 100.0
        is_flagged = scorer.get_manual_review_flag(conf_percent)
        if is_flagged:
            flag_count += 1
        extracted_fields[e.entity_type].append({
            "value": e.entity_value,
            "confidence_score": round(conf_percent, 2),
            "requires_review": is_flagged,
        })
        confidence_tally += conf_percent

    overall_confidence = (confidence_tally / len(entities)) if entities else 0.0

    return {
        "document_id": document_id,
        "status": "processed",
        "document_type": extraction.document_class,
        "detected_language": extraction.detected_language,
        "global_confidence": round(overall_confidence, 2),
        "manual_review_required": (overall_confidence < 75.0) or (flag_count > 0),
        "flagged_fields": flag_count,
        "processing_time_seconds": round(extraction.processing_time_seconds, 2),
        "extracted_fields": extracted_fields,
        "raw_text_preview": (extraction.raw_text or "")[:200],
    }


@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_stats(db: Session = Depends(get_db)):
    """Real-time pipeline statistics including per-language document breakdown."""
    req_id = _request_id_ctx.get("-")

    try:
        total_docs = db.query(Document).count()
        total_extractions = db.query(Extraction).count()

        avg_time_result = db.query(func.avg(Extraction.processing_time_seconds)).scalar()
        avg_time = round(avg_time_result, 3) if avg_time_result else 0.0

        avg_conf_result = db.query(func.avg(Entity.confidence_score)).scalar()
        avg_conf = round(avg_conf_result * 100.0, 2) if avg_conf_result else 0.0

        lang_rows = (
            db.query(Extraction.detected_language, func.count(Extraction.id))
            .filter(Extraction.detected_language.isnot(None))
            .group_by(Extraction.detected_language)
            .all()
        )
        language_breakdown = {lang: count for lang, count in lang_rows}

    except SQLAlchemyError as exc:
        logger.error(f"[{req_id}] DB error fetching stats: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database error while retrieving statistics.")

    return {
        "system_health": "online",
        "statistics": {
            "total_documents_uploaded": total_docs,
            "total_documents_analyzed": total_extractions,
            "pipeline_average_time_seconds": avg_time,
            "system_average_confidence_percentage": avg_conf,
            "documents_by_language": language_breakdown,
        },
    }
