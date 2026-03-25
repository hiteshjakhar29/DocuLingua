import os
import json
import logging
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm

import fitz  # PyMuPDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Optional: Add project root to sys path if run from root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ml_models.ocr import OCREngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainClassifier")

class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def process_pdfs_to_texts(metadata_path: str, cache_path: str) -> Tuple[List[str], List[str]]:
    """
    Extracts text from all PDFs in the metadata using OCREngine.
    Uses a cache file to avoid re-running intensive OCR if script restarts.
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    ocr_cache = {}
    if os.path.exists(cache_path):
        logger.info(f"Loading cached OCR results from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            ocr_cache = json.load(f)
            
    engine = OCREngine()
    base_dir = os.path.dirname(metadata_path)
    
    texts = []
    labels = []
    
    logger.info("Starting text extraction phase...")
    updated_cache = False
    
    for filename, data in tqdm(metadata.items(), desc="Processing PDFs"):
        doc_type = data['document_type']
        language = data['language']
        # Map Faker locale back to 2-letter lang code used by OCREngine
        lang_code = language.split('_')[0] 
        
        pdf_path = os.path.join(base_dir, doc_type, filename)
        
        if filename in ocr_cache:
            extracted_text = ocr_cache[filename]
        else:
            if not os.path.exists(pdf_path):
                logger.warning(f"File missing: {pdf_path}. Skipping.")
                continue
                
            # Render first page of PDF to an image for OCR
            try:
                doc = fitz.open(pdf_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=200)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    img_path = tmp.name
                    pix.save(img_path)
                    
                # Run complete Dual OCR
                result = engine.run_dual_ocr(img_path, langs=[lang_code])
                os.remove(img_path)
                doc.close()
                
                extracted_text = result.get('extracted_text', "")
                ocr_cache[filename] = extracted_text
                updated_cache = True
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue
                
        texts.append(extracted_text)
        labels.append(doc_type)

    if updated_cache:
        logger.info(f"Saving new OCR results to cache at {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_cache, f, ensure_ascii=False)
            
    return texts, labels

def main():
    metadata_path = "data/synthetic_docs/metadata.json"
    cache_path = "data/synthetic_docs/ocr_cache.json"
    model_output_dir = "models/classifier"
    results_dir = "results"
    
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("Step 1: Extracing and Caching Document Texts")
    texts, labels = process_pdfs_to_texts(metadata_path, cache_path)
    
    if not texts:
        logger.error("No texts extracted. Exiting.")
        return

    logger.info(f"Loaded {len(texts)} documents successfully.")
    
    # Label Encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    class_names = list(label_encoder.classes_)
    logger.info(f"Unique classes identified: {class_names}")

    logger.info("Step 2: Splitting Dataset (70% Train, 15% Val, 15% Test)")
    # Train = 70%, Temp = 30%
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, encoded_labels, test_size=0.30, stratify=encoded_labels, random_state=42
    )
    # Val = 15%, Test = 15% (Half of Temp)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    logger.info(f"Splits - Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    logger.info("Step 3: Initializing Tokenizer and Model")
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(class_names))
    
    # Save label mappings to model config
    model.config.id2label = {idx: label for idx, label in enumerate(class_names)}
    model.config.label2id = {label: idx for idx, label in enumerate(class_names)}
    
    logger.info("Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = DocumentDataset(train_encodings, train_labels)
    val_dataset = DocumentDataset(val_encodings, val_labels)
    test_dataset = DocumentDataset(test_encodings, test_labels)

    logger.info("Step 4: Configuring TrainingArguments")
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    logger.info("Step 5: Beginning Fine-Tuning")
    try:
        trainer.train()
    finally:
        logger.info(f"Saving final model and tokenizer to {model_output_dir}")
        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        # Save label mappings so the model is self-contained
        label_map = {"id2label": model.config.id2label, "label2id": model.config.label2id}
        with open(os.path.join(model_output_dir, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)

    logger.info("Step 6: Evaluating on Test Set")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = test_labels
    
    # Per-class metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
    overall_acc = accuracy_score(y_true, y_pred)
    
    logger.info(f"Overall Test Accuracy: {overall_acc:.4f} (Target: >0.89)")
    
    # Classification Report
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n--- Classification Report ---\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Compile Results
    evaluation_results = {
        "overall_accuracy": overall_acc,
        "per_class_metrics": {},
        "classification_report": cr
    }
    
    for i, class_name in enumerate(class_names):
        evaluation_results["per_class_metrics"][class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(fscore[i]),
            "support": int(support[i])
        }

    # Save Metrics JSON
    eval_path = os.path.join(results_dir, "classifier_evaluation.json")
    with open(eval_path, "w", encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4)
        
    logger.info(f"Saved evaluation metrics to {eval_path}")

    # Step 7: Save Confusion Matrix PNG
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Document Classification Confusion Matrix')
    plt.tight_layout()
    
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved confusion matrix to {cm_path}")
    logger.info("Pipeline Complete. Model and metrics saved.")

def promote_best_checkpoint(model_output_dir: str = "models/classifier"):
    """
    Copy the best (lowest-loss) checkpoint to models/classifier/ as the final model.
    Run this if training was interrupted before save_model() executed.
    """
    import shutil
    import re

    checkpoints = [
        d for d in os.listdir(model_output_dir)
        if re.match(r"checkpoint-\d+", d)
    ]
    if not checkpoints:
        print("No checkpoints found.")
        return

    # trainer_state.json inside each checkpoint records best_metric
    best_ckpt, best_loss = None, float("inf")
    for ckpt in checkpoints:
        state_path = os.path.join(model_output_dir, ckpt, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            loss = state.get("best_metric", float("inf"))
            if loss < best_loss:
                best_loss, best_ckpt = loss, ckpt

    if best_ckpt is None:
        best_ckpt = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        print(f"Could not determine best by loss; using latest: {best_ckpt}")
    else:
        print(f"Best checkpoint: {best_ckpt} (loss={best_loss:.4f})")

    src = os.path.join(model_output_dir, best_ckpt)
    for fname in os.listdir(src):
        shutil.copy2(os.path.join(src, fname), os.path.join(model_output_dir, fname))
    print(f"Promoted {best_ckpt} → {model_output_dir}")


if __name__ == "__main__":
    import sys
    if "--promote-checkpoint" in sys.argv:
        promote_best_checkpoint()
    else:
        main()
