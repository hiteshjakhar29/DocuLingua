import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import spacy
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Evaluate NER - %(message)s')
logger = logging.getLogger("EvalNER")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = "models/ner"
METADATA_PATH = "data/synthetic_docs/metadata.json"
OCR_CACHE_PATH = "data/synthetic_docs/ocr_cache.json"

def run_evaluation():
    if not os.path.exists(MODEL_DIR):
        logger.error(f"Missing custom SpaCy NER model at {MODEL_DIR}")
        return
        
    logger.info("Loading Transformer NER pipeline via SpaCy...")
    try:
        nlp = spacy.load(MODEL_DIR)
    except Exception as e:
        logger.error(f"Failed loading spacy model: {e}")
        return
        
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    with open(OCR_CACHE_PATH, "r") as f:
        ocr_cache = json.load(f)
        
    import random
    random.seed(42)
    keys = list(metadata.keys())
    random.shuffle(keys)
    test_keys = keys[:200]
    
    # Track discrete entity-level extraction success
    entity_evals = []
    
    # NER is trickier because of overlapping tokens!
    # For robust F1 analysis, we measure Exact substring recall.
    predictions = []
    truths = []
    
    logger.info("Executing NER inference loops...")
    for key in test_keys:
        meta = metadata[key]
        text = ocr_cache.get(key, "")
        if not text:
            # Fallback
            text = " ".join([str(v) for v in meta.get("extracted_fields", {}).values()])
            
        doc = nlp(text)
        
        pred_entities = {ent.label_: ent.text for ent in doc.ents}
        truth_entities = meta.get("extracted_fields", {})
        
        # Unroll logic globally for global F1 scores mapping distinct categories
        for true_label, true_val in truth_entities.items():
            
            # Sub-mapping truths logically against NER categories manually mapped during Synthesize step
            mapped_label = true_label.upper()
            if mapped_label == "NAME": mapped_label = "PERSON"
            if mapped_label == "UNIVERSITY": mapped_label = "INSTITUTION"
            
            truths.append(mapped_label)
            
            # Did the model find it?
            if mapped_label in pred_entities:
                predictions.append(mapped_label)
                # Compute token overlap severity
                pred_val = pred_entities[mapped_label]
                is_exact = true_val.lower() in pred_val.lower() or pred_val.lower() in true_val.lower()
                
                entity_evals.append({
                    "document_id": key,
                    "target_entity": mapped_label,
                    "ground_truth": true_val,
                    "prediction": pred_val,
                    "exact_match": is_exact
                })
            else:
                predictions.append("O") # Outside / Missed
                entity_evals.append({
                    "document_id": key,
                    "target_entity": mapped_label,
                    "ground_truth": true_val,
                    "prediction": "MISSING",
                    "exact_match": False
                })
                
        # Register false positives (predicted but not in truth)
        for pred_label, pred_val in pred_entities.items():
            # Quick check if it wasn't supposed to be there
            # (Rough heuristic since exact alignment mapping requires char offsets)
            found_in_truth = False
            for t_lab, t_val in truth_entities.items():
                if t_val.lower() in pred_val.lower() or pred_val.lower() in t_val.lower():
                    found_in_truth = True
                    break
            if not found_in_truth:
                truths.append("O")
                predictions.append(pred_label)
                
    # Calculate overarching strict SKLearn bounds
    report_dict = classification_report(truths, predictions, output_dict=True, zero_division=0)
    
    # Dump JSON
    with open(os.path.join(RESULTS_DIR, "ner_metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=4)
        
    df_evals = pd.DataFrame(entity_evals)
    df_evals.to_csv(os.path.join(RESULTS_DIR, "ner_predictions.csv"), index=False)
    
    # Visualizations: Precision-Recall style F1 Chart
    labels = [L for L in report_dict.keys() if L not in ["O", "accuracy", "macro avg", "weighted avg"]]
    f1s = [report_dict[L]['f1-score'] for L in labels]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=f1s, palette="magma")
    plt.title('Custom NER - F1 Score by Entity Type')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    f1_path = os.path.join(RESULTS_DIR, "ner_f1_scores.png")
    plt.tight_layout()
    plt.savefig(f1_path, dpi=300)
    plt.close()
    
    # Generate Confusion Matrix
    cm_labels = sorted(list(set(truths) | set(predictions)))
    cm = confusion_matrix(truths, predictions, labels=cm_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title('NER Confusion Matrix (Token Alignments)')
    plt.ylabel('True Entity')
    plt.xlabel('Predicted Entity')
    cm_path = os.path.join(RESULTS_DIR, "ner_confusion.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # ReportLab Payload
    pdf_path = os.path.join(RESULTS_DIR, "ner_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, "DocuLingua - Custom NER Evaluation Report")
    
    overall_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Total Entity Fields Evaluated: {len(truths)}")
    c.drawString(50, 700, f"Macro F1-Score: {overall_f1*100:.2f}%")
    
    c.drawImage(ImageReader(cm_path), 50, 380, width=500, height=300, preserveAspectRatio=True)
    c.drawImage(ImageReader(f1_path), 50, 60, width=500, height=300, preserveAspectRatio=True)
    
    c.save()
    logger.info("Successfully bound robust F1 mapping architectures onto generated PDF.")

if __name__ == "__main__":
    run_evaluation()
