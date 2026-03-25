import os
import json
import random
import re
import logging
import spacy
from spacy.training import Example
from spacy.util import minibatch
from spacy.scorer import Scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainNER")

ENTITY_MAP = {
    "name": "PERSON",
    "institution": "INSTITUTION",
    "profession": "DEGREE",
    "reason": "CERT_NUMBER", # Fallback proxy for this entity
    "date": "DATE",
    "valid_until": "DATE"
}

def load_data(metadata_path, ocr_cache_path):
    """
    Parses OCR-extracted text and maps metadata ground truths onto their character offsets.
    """
    logger.info("Loading ground truth metadata and OCR cache...")
    if not os.path.exists(ocr_cache_path):
        raise FileNotFoundError(f"Missing OCR Cache at {ocr_cache_path}. Run classifier script first to generate OCR.")
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(ocr_cache_path, 'r', encoding='utf-8') as f:
        ocr_cache = json.load(f)

    training_data = []
    
    for filename, meta in metadata.items():
        if filename not in ocr_cache:
            continue
            
        text = ocr_cache[filename]
        entities = []
        extracted_fields = meta.get("extracted_fields", {})
        
        # Exact match alignment
        for key, val in extracted_fields.items():
            if not val or key not in ENTITY_MAP:
                continue
            
            label = ENTITY_MAP[key]
            start = text.find(val)
            if start != -1:
                end = start + len(val)
                entities.append((start, end, label))
                
        # Regex heuristics for GRADES (A, B, C, A+, B+)
        if meta.get("document_type") == "transcript":
            for match in re.finditer(r'\b([A-C]\+?)\b', text):
                start, end = match.span()
                overlap = any(s < end and e > start for (s, e, _) in entities)
                if not overlap:
                    entities.append((start, end, "GRADE"))
                    break # Cap it to 1 grade sample per transcript to avoid noise

        # Resolve bounds
        entities = sorted(entities, key=lambda x: x[0])
        resolved_entities = []
        last_end = -1
        for start, end, label in entities:
             if start >= last_end:
                 resolved_entities.append((start, end, label))
                 last_end = end
                 
        if resolved_entities:
            training_data.append((text, {"entities": resolved_entities}))
            
    logger.info(f"Successfully aligned and processed {len(training_data)} training examples.")
    return training_data

def evaluate_model(nlp, test_data, results_dir):
    """
    Generates Precision/Recall/F1 metrics and outputs a Confusion Matrix.
    """
    logger.info("Evaluating model on Test Set...")
    scorer = Scorer()
    examples = []
    
    y_true = []
    y_pred = []
    
    report_examples = []
    
    for text, annot in test_data:
        doc_true = nlp.make_doc(text)
        example = Example.from_dict(doc_true, annot)
        
        doc_pred = nlp(text)
        example.predicted = doc_pred
        examples.append(example)
        
        # Token-level arrays for Confusion Matrix & Sklearn metrics
        for i in range(len(doc_pred)):
            p_label = doc_pred[i].ent_type_ if doc_pred[i].ent_type_ else "O"
            t_label = example.reference[i].ent_type_ if example.reference[i].ent_type_ else "O"
            
            # Subsample non-entities to prevent massive 'O' class bias
            if p_label != "O" or t_label != "O" or random.random() < 0.05:
                y_true.append(t_label)
                y_pred.append(p_label)
                
        # Save a sample of correct/incorrect for reporting
        if len(report_examples) < 20 and len(doc_pred.ents) > 0:
            correct = [ent.text for ent in doc_pred.ents if ent.label_ in [e[2] for e in annot['entities']]]
            report_examples.append({
                "text_snippet": text[:100].replace('\n', ' ') + "...",
                "predicted_entities": [(ent.text, ent.label_) for ent in doc_pred.ents],
                "true_entities": [(text[s:e], label) for s, e, label in annot['entities']]
            })
            
    # SpaCy metrics
    scores = scorer.score(examples)
    overall_f1 = scores.get('ents_f', 0.0)
    logger.info(f"SpaCy Scorer Overall F1: {overall_f1 * 100:.2f}% (Target: >87%)")
    
    # Sklearn Class-wise metrics
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print("\n--- NER Classification Report ---\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Save Report
    eval_path = os.path.join(results_dir, "ner_evaluation.json")
    with open(eval_path, "w", encoding='utf-8') as f:
        json.dump({
            "overall_f1_score": overall_f1,
            "spacy_detailed_scores": scores,
            "sklearn_classification_report": cr
        }, f, indent=4)
        
    logger.info(f"Saved evaluation metrics to {eval_path}")
    
    # Confusion Matrix
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('NER Entity Confusion Matrix')
    plt.tight_layout()
    
    cm_path = os.path.join(results_dir, "ner_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")
    
    # Final textual example report
    examples_path = os.path.join(results_dir, "examples_report.txt")
    with open(examples_path, "w", encoding='utf-8') as f:
        for i, ex in enumerate(report_examples):
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Text Snippet: {ex['text_snippet']}\n")
            f.write(f"Predicted: {ex['predicted_entities']}\n")
            f.write(f"Actual: {ex['true_entities']}\n\n")

def main():
    metadata_path = "data/synthetic_docs/metadata.json"
    ocr_cache_path = "data/synthetic_docs/ocr_cache.json"
    results_dir = "results"
    model_output_dir = "models/ner"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 1. Generate annotated data
    data = load_data(metadata_path, ocr_cache_path)
    if not data:
        logger.error("No training data found. Exiting.")
        return
        
    train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
    logger.info(f"Split data into {len(train_data)} train and {len(test_data)} test items.")
    
    # 2. Setup SpaCy models
    model_name = "en_core_web_sm"
    logger.info(f"Loading Base Model: {model_name}")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.error(f"Model '{model_name}' not found. Please install via: python -m spacy download {model_name}")
        return
        
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
        
    # Add custom labels
    for text, annot in train_data:
        for ent in annot.get("entities"):
            ner.add_label(ent[2])
            
    logger.info("Labels initialized. Beginning execution...")

    # 3. Training execution
    epochs = 30
    batch_size = 8
    dropout = 0.2
    
    # Freeze other pipes to strictly update NER
    pipe_exceptions = ["ner"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    with nlp.disable_pipes(*unaffected_pipes):
        # spaCy 3.7+: resume_training() was removed.
        # initialize() wires up new label vectors while keeping pretrained
        # transformer weights intact; create_optimizer() then returns an Adam
        # optimizer without resetting any existing weights.
        def get_examples():
            return [Example.from_dict(nlp.make_doc(t), a) for t, a in train_data[:10]]

        nlp.initialize(get_examples=get_examples)
        optimizer = nlp.create_optimizer()
        
        for iteration in range(epochs):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=batch_size)
            
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    examples.append(Example.from_dict(doc, annotations[i]))
                    
                nlp.update(
                    examples, 
                    drop=dropout, 
                    sgd=optimizer, 
                    losses=losses
                )
            logger.info(f"Epoch {iteration + 1} / {epochs} - Losses: {losses}")

    # 4. Evaluation
    evaluate_model(nlp, test_data, results_dir)
    
    # 5. Save model
    nlp.to_disk(model_output_dir)
    logger.info(f"Saved fine-tuned custom NER model to {model_output_dir}")

if __name__ == "__main__":
    main()
