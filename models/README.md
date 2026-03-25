# Models

This directory holds the two trained ML models used in the DocuLingua pipeline.

```
models/
├── classifier/          # Fine-tuned multilingual BERT (document type classifier)
│   └── .gitkeep         # Weights excluded from git — see note below
└── ner/                 # Fine-tuned spaCy NER pipeline (~15 MB, committed)
    ├── ner/             # NER component weights
    ├── tok2vec/         # Shared token-to-vector encoder
    ├── config.cfg       # Full pipeline config
    ├── meta.json        # Model metadata and labels
    └── ...              # Other spaCy pipeline components
```

> **Note:** Trained classifier weights are not included in this repository due to size (6 GB+).
> Run the training commands below to produce them locally.
> The spaCy NER model (~15 MB) is committed and ready to use.

---

## 1. Document Classifier (`models/classifier/`)

### Architecture

| Property        | Value                                              |
|-----------------|----------------------------------------------------|
| Base model      | `bert-base-multilingual-cased` (HuggingFace)       |
| Task            | Sequence classification                            |
| Output classes  | 6                                                  |
| Max input length| 512 tokens                                         |
| Framework       | PyTorch + HuggingFace Transformers                 |

**Classes:**

| ID | Label                  |
|----|------------------------|
| 0  | `certificate`          |
| 1  | `diploma`              |
| 2  | `employment_letter`    |
| 3  | `professional_license` |
| 4  | `transcript`           |
| 5  | `university_degree`    |

### Training Details

| Hyperparameter          | Value              |
|-------------------------|--------------------|
| Epochs (max)            | 5                  |
| Batch size (train/eval) | 16 / 16            |
| Learning rate           | 2e-5               |
| Weight decay            | 0.01               |
| Optimizer               | AdamW              |
| Early stopping patience | 1 epoch            |
| Data split              | 70% train / 15% val / 15% test |
| Best checkpoint         | step 220 (epoch 5) |
| Best eval loss          | 0.01714            |

**Training progression (validation set):**

| Epoch | Accuracy | F1     | Loss   |
|-------|----------|--------|--------|
| 1     | 96.00%   | 95.96% | 0.3993 |
| 2     | 98.67%   | 98.65% | 0.0786 |
| 3     | 99.33%   | 99.33% | 0.0289 |
| 4     | 100.00%  | 100%   | 0.0176 |
| 5     | 100.00%  | 100%   | 0.0171 |

### Performance (held-out test set, 151 documents)

| Metric   | Target | Achieved  |
|----------|--------|-----------|
| Accuracy | 89%    | **100%**  |
| F1       | —      | **100%**  |

All 6 classes: precision 1.00 / recall 1.00 / F1 1.00.
See [`results/classifier_evaluation.json`](../results/classifier_evaluation.json) and
[`results/confusion_matrix.png`](../results/confusion_matrix.png).

### File Inventory

| File / Directory       | Size    | Purpose                                                        |
|------------------------|---------|----------------------------------------------------------------|
| `model.safetensors`    | 679 MB  | Final model weights (safe tensors format)                      |
| `config.json`          | 4 KB    | Model config + `id2label` / `label2id` mappings                |
| `checkpoint-176/`      | 2.0 GB  | Epoch 4 checkpoint (100% validation accuracy)                  |
| `checkpoint-220/`      | 2.0 GB  | Epoch 5 checkpoint — selected as best by lowest eval loss      |
| `optimizer.pt`         | 1.3 GB  | Adam optimizer state (needed to resume training only)          |
| `scheduler.pt`         | 4 KB    | Learning-rate scheduler state                                  |
| `rng_state.pth`        | 16 KB   | RNG snapshot for reproducibility                               |
| `trainer_state.json`   | 8 KB    | Full training log (loss, metrics per step)                     |
| `training_args.bin`    | 8 KB    | Serialized `TrainingArguments`                                 |

> `optimizer.pt`, `rng_state.pth`, and `scheduler.pt` are only needed to **resume** an
> interrupted training run. They are not required for inference.

---

## 2. NER Model (`models/ner/`)

### Architecture

| Property        | Value                                              |
|-----------------|----------------------------------------------------|
| Base model      | `en_core_web_sm` v3.7.1 (spaCy / Explosion AI)    |
| Training method | Fine-tuned NER component; other pipes frozen       |
| Custom labels   | 6 (see below)                                      |
| Framework       | spaCy 3.7+                                         |

**Custom entity labels:**

| Label         | Description                                    | Example                            |
|---------------|------------------------------------------------|------------------------------------|
| `PERSON`      | Name of the certificate holder                 | `Élise Fontaine`                   |
| `INSTITUTION` | Issuing school, university, or organisation    | `Normand Nguyen S.A. Academy`      |
| `DEGREE`      | Qualification or professional title            | `Bachelor of Science`              |
| `DATE`        | Issue or expiry date (ISO format)              | `2024-05-15`                       |
| `CERT_NUMBER` | Certificate or license identifier              | `LIC-2024-00187`                   |
| `GRADE`       | Academic grade (transcript documents only)     | `A+`                               |

### Training Details

| Hyperparameter | Value                     |
|----------------|---------------------------|
| Epochs         | 30                        |
| Batch size     | 8                         |
| Dropout        | 0.2                       |
| Data split     | 85% train / 15% test      |
| Pipes frozen   | tok2vec, tagger, parser, lemmatizer, attribute_ruler |

Ground truth was produced by exact-string alignment of OCR output against the
metadata generated by `data/synthesize.py`. GRADE entities were matched via regex
(`\b[A-C]\+?\b`) on transcript documents.

### Performance (spaCy entity scorer, held-out test set)

| Metric         | Target | Achieved     |
|----------------|--------|--------------|
| Overall F1     | 87%    | **96.22%**   |
| Precision      | —      | **94.68%**   |
| Recall         | —      | **97.80%**   |

Per-entity breakdown:

| Entity        | Precision | Recall | F1       | Notes                                     |
|---------------|-----------|--------|----------|-------------------------------------------|
| `DATE`        | 1.000     | 1.000  | **1.000**| Perfect — ISO format is unambiguous       |
| `DEGREE`      | 1.000     | 1.000  | **1.000**| Clean pattern in training data            |
| `PERSON`      | 0.933     | 1.000  | **0.966**| Minor false positives on lowercase names  |
| `INSTITUTION` | 0.727     | 0.800  | **0.762**| Hardest class — variable name formats     |

See [`results/ner_evaluation.json`](../results/ner_evaluation.json) and
[`results/ner_confusion_matrix.png`](../results/ner_confusion_matrix.png).

### File Inventory

| File / Directory    | Size   | Purpose                                              |
|---------------------|--------|------------------------------------------------------|
| `ner/`              | 5.9 MB | NER component weights (transition system + model)    |
| `tok2vec/`          | 6.0 MB | Shared token encoder weights                         |
| `vocab/`            | 1.2 MB | Vocabulary, string store, and lexeme data             |
| `lemmatizer/`       | 952 KB | Lemmatizer lookup tables                             |
| `parser/`           | 324 KB | Dependency parser weights (frozen, not retrained)    |
| `senter/`           | 200 KB | Sentence segmenter (disabled at runtime)             |
| `tagger/`           | 24 KB  | POS tagger config (frozen)                           |
| `attribute_ruler/`  | 16 KB  | Attribute rewrite rules                              |
| `tokenizer`         | 76 KB  | Tokenizer exception data                             |
| `config.cfg`        | 8 KB   | Full spaCy pipeline configuration                    |
| `meta.json`         | 12 KB  | Model metadata, label list, base model performance   |

---

## 3. Retraining

### Prerequisites

```bash
# Ensure the virtual environment is active and deps are installed
pip install -r requirements.txt

# Validate environment variables are set
python validate_env.py
```

### Step 1 — Generate synthetic training data

```bash
python data/synthesize.py
# Output: data/synthetic_docs/  (PDFs + metadata.json)
```

### Step 2 — Train the document classifier

```bash
python backend/ml_models/train_classifier.py
# Reads:   data/synthetic_docs/metadata.json
# Writes:  models/classifier/  (weights, config, checkpoints)
#          results/classifier_evaluation.json
#          results/confusion_matrix.png
#
# Runtime: ~20 min on CPU, ~5 min on GPU
# Disk:    ~6 GB for weights + 2 checkpoints
```

If training is interrupted before the final `save_model()` call, promote the
best checkpoint manually:

```bash
python backend/ml_models/train_classifier.py --promote-checkpoint
```

### Step 3 — Train the NER pipeline

```bash
# Requires the OCR cache produced in Step 2
python backend/ml_models/train_ner.py
# Reads:   data/synthetic_docs/metadata.json
#          data/synthetic_docs/ocr_cache.json  (generated during classifier training)
# Writes:  models/ner/
#          results/ner_evaluation.json
#          results/ner_confusion_matrix.png
#          results/examples_report.txt
#
# Runtime: ~5 min on CPU
# Disk:    ~15 MB
```

### Step 4 — Evaluate independently (optional)

```bash
python backend/evaluate_classifier.py   # Confusion matrix + per-class metrics
python backend/evaluate_ner.py          # Token-level F1 / precision / recall
python backend/evaluate_ab_test.py      # Benchmark against baseline
```

### Full automated setup (recommended)

```bash
./quick_setup.sh              # Runs all 8 steps including DB init and model training
./quick_setup.sh --skip-train # Skip training and use graceful model degradation
```

---

## 4. Using the Models in Inference

Both models are loaded at API startup (`backend/main.py`) and are optional — the
pipeline degrades gracefully if either is absent:

- **No classifier model** → document type returned as `unknown`
- **No NER model** → entity extraction skipped, confidence score reduced

To load them directly in Python:

```python
# Classifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("models/classifier")
model = AutoModelForSequenceClassification.from_pretrained("models/classifier")
inputs = tokenizer("University Degree awarded to ...", return_tensors="pt", truncation=True)
logits = model(**inputs).logits
label_id = torch.argmax(logits, dim=-1).item()
print(model.config.id2label[label_id])

# NER
import spacy
nlp = spacy.load("models/ner")
doc = nlp("This certifies that Jane Smith has completed...")
for ent in doc.ents:
    print(ent.text, ent.label_)
```
