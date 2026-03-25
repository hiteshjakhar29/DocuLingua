import json
import os
import random
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - Evaluate Classifier - %(message)s"
)
logger = logging.getLogger("EvalClassifier")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = "models/classifier"
METADATA_PATH = "data/synthetic_docs/metadata.json"
OCR_CACHE_PATH = "data/synthetic_docs/ocr_cache.json"


def run_evaluation() -> None:
    # ------------------------------------------------------------------
    # 1. Load classifier model
    # ------------------------------------------------------------------
    if not os.path.exists(MODEL_DIR):
        logger.error(
            f"No trained model found at '{MODEL_DIR}'. "
            "Run quick_setup.sh to train the classifier first."
        )
        return

    try:
        classifier = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            max_length=512,
            truncation=True,
        )
        logger.info(f"Loaded classifier from {MODEL_DIR}")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error(f"Failed to load classifier pipeline: {exc}")
        return

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    if not os.path.exists(METADATA_PATH):
        logger.error(f"Metadata file not found at '{METADATA_PATH}'. Run data/synthesize.py first.")
        return
    if not os.path.exists(OCR_CACHE_PATH):
        logger.error(f"OCR cache not found at '{OCR_CACHE_PATH}'. Run train_classifier.py first.")
        return

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    with open(OCR_CACHE_PATH, "r") as f:
        ocr_cache = json.load(f)

    random.seed(42)
    keys = list(metadata.keys())
    random.shuffle(keys)
    test_keys = keys[:200]

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    misclassified: list[dict] = []

    logger.info("Running inference on test set…")

    for key in test_keys:
        meta = metadata[key]
        true_class = meta.get("document_type", "unknown")

        # Use OCR cache text; fall back to concatenated field values if empty
        text = ocr_cache.get(key, "").strip()
        if len(text) < 10:
            text = " ".join(str(v) for v in meta.get("extracted_fields", {}).values())

        if not text.strip():
            logger.debug(f"Skipping '{key}': no usable text.")
            continue

        try:
            result = classifier(text[:2000])

            # pipeline() with a classification model returns a list of dicts;
            # the first element is the top prediction.
            if not result or not isinstance(result, list):
                raise ValueError(f"Unexpected pipeline output: {result!r}")

            top = result[0]
            if not isinstance(top, dict) or "label" not in top or "score" not in top:
                raise ValueError(f"Malformed prediction dict: {top!r}")

            pred_class = top["label"]
            score = float(top["score"])

        except (ValueError, RuntimeError, TypeError) as exc:
            logger.warning(f"Inference failed for '{key}': {exc}")
            pred_class = "unknown"
            score = 0.0

        y_true.append(true_class)
        y_pred.append(pred_class)
        confidences.append(score)

        if true_class != pred_class:
            misclassified.append({
                "document_id": key,
                "true_class": true_class,
                "predicted_class": pred_class,
                "confidence": score,
                "text_snippet": text[:100].replace("\n", " ") + "…",
            })

    # ------------------------------------------------------------------
    # 4. Edge-case guards
    # ------------------------------------------------------------------
    if not y_true:
        logger.error("No documents were successfully evaluated. Aborting.")
        return

    if len(set(y_pred)) == 1:
        logger.warning(
            f"All predictions are the same class ('{y_pred[0]}'). "
            "The model may not have been fine-tuned correctly."
        )

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    logger.info(f"Overall accuracy: {acc * 100:.2f}% on {len(y_true)} documents")
    print(classification_report(y_true, y_pred, zero_division=0))

    metrics_payload = {
        "total_evaluated": len(y_true),
        "overall_accuracy": round(acc, 4),
        "classification_report": report_dict,
    }
    with open(os.path.join(RESULTS_DIR, "classifier_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=4)

    if misclassified:
        pd.DataFrame(misclassified).to_csv(
            os.path.join(RESULTS_DIR, "classifier_errors.csv"), index=False
        )
        logger.info(f"Wrote {len(misclassified)} misclassification examples to classifier_errors.csv")

    # ------------------------------------------------------------------
    # 6. Confusion matrix
    # ------------------------------------------------------------------
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Classifier Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "classifier_confusion.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # 7. F1 bar chart
    # ------------------------------------------------------------------
    f1_scores = [report_dict[L]["f1-score"] for L in labels if L in report_dict]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=f1_scores, palette="viridis")
    plt.title("F1 Score by Document Type")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    f1_path = os.path.join(RESULTS_DIR, "classifier_f1_scores.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # 8. PDF report
    # ------------------------------------------------------------------
    pdf_path = os.path.join(RESULTS_DIR, "classifier_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, "DocuLingua — Classifier Evaluation Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Documents evaluated: {len(y_true)}")
    c.drawString(50, 700, f"Overall accuracy: {acc * 100:.2f}%")
    c.drawImage(ImageReader(cm_path), 50, 420, width=500, height=270, preserveAspectRatio=True)
    c.drawImage(ImageReader(f1_path), 50, 120, width=500, height=270, preserveAspectRatio=True)
    c.save()

    logger.info("Evaluation complete. Outputs saved to results/")


if __name__ == "__main__":
    run_evaluation()
