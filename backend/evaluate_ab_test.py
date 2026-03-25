import os
import json
import time
import random
import difflib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import fitz  # PyMuPDF
import tempfile

# Force discovery of backend module
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.ml_models.ocr import OCREngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - A/B Test - %(levelname)s - %(message)s')
logger = logging.getLogger("AB_Testing")


def compute_accuracy(truth_strings, extracted_text):
    """
    Computes character and word accuracy by measuring how well ground-truth
    field values survived the OCR pipeline.

    For each ground-truth string:
      - Score 1.0 if it appears verbatim (case-insensitive) in the OCR output.
      - Otherwise use the longest contiguous matching block (SequenceMatcher)
        as the character score, and word-presence ratio as the word score.

    Returns (avg_char_accuracy, avg_word_accuracy), both in [0, 1].
    """
    if not truth_strings:
        return 1.0, 1.0  # nothing to measure → treat as perfect

    char_ratios = []
    word_ratios = []
    search_space = extracted_text.lower()

    for truth in truth_strings:
        if not truth:
            continue
        truth_lower = str(truth).lower()

        if truth_lower in search_space:
            char_ratios.append(1.0)
            word_ratios.append(1.0)
        else:
            seq = difflib.SequenceMatcher(None, truth_lower, search_space)
            match = seq.find_longest_match(0, len(truth_lower), 0, len(search_space))
            c_score = match.size / len(truth_lower)
            char_ratios.append(c_score)

            truth_words = truth_lower.split()
            if truth_words:
                found_words = sum(1 for w in truth_words if w in search_space)
                w_score = found_words / len(truth_words)
            else:
                w_score = 1.0
            word_ratios.append(w_score)

    avg_char = sum(char_ratios) / len(char_ratios)
    avg_word = sum(word_ratios) / len(word_ratios)
    return avg_char, avg_word


def run_evaluation(sample_size=200):
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    METADATA_PATH = "data/synthetic_docs/metadata.json"
    if not os.path.exists(METADATA_PATH):
        logger.error(f"Cannot run A/B test without synthetic ground truth JSON at {METADATA_PATH}.")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Stratified sample — fixed seed for reproducibility
    candidates = list(metadata.items())
    random.seed(42)
    random.shuffle(candidates)
    test_docs = candidates[:min(sample_size, len(candidates))]
    logger.info(f"Loaded {len(test_docs)} documents for paired A/B evaluation...")

    ocr = OCREngine()
    experiment_results = []

    for i, (filename, meta) in enumerate(test_docs):
        doc_type = meta.get("document_type", "certificate")
        doc_path = os.path.join("data/synthetic_docs", doc_type, filename)

        if not os.path.exists(doc_path):
            continue

        noise_level = meta.get("noise_applied", False)
        lang = meta.get("language", "en")
        truths = list(meta.get("extracted_fields", {}).values())

        if lang == "en" and not noise_level:
            condition = "Clean English"
        elif lang == "en" and noise_level:
            condition = "Noisy English"
        elif lang != "en" and not noise_level:
            condition = "Clean Multilingual"
        else:
            condition = "Noisy Multilingual"

        tmp_img = None
        try:
            pdf_doc = fitz.open(doc_path)
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_img = tmp.name
                pix.save(tmp_img)
            pdf_doc.close()

            # ---- A: TESSERACT BASELINE ----
            t0 = time.time()
            tess_text = ocr.extract_text_tesseract(ocr.preprocess_image(tmp_img), [lang])
            tess_time = time.time() - t0
            tess_char, tess_word = compute_accuracy(truths, tess_text.get("text", ""))

            # ---- B: DUAL ENGINE (Tesseract + EasyOCR) ----
            t1 = time.time()
            dual_results = ocr.run_dual_ocr(tmp_img, [lang])
            dual_text = dual_results.get("extracted_text", "")
            dual_time = time.time() - t1
            dual_char, dual_word = compute_accuracy(truths, dual_text)

            # No artificial adjustments — record raw measured values only.
            experiment_results.append({
                "document_id": filename,
                "language": lang,
                "is_noisy": noise_level,
                "condition_group": condition,
                "tesseract_char_acc": tess_char,
                "tesseract_word_acc": tess_word,
                "tesseract_time": tess_time,
                "dual_char_acc": dual_char,
                "dual_word_acc": dual_word,
                "dual_time": dual_time,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1} / {len(test_docs)} documents...")

        except Exception as e:
            logger.warning(f"Error evaluating {filename}: {e}")
            continue
        finally:
            if tmp_img and os.path.exists(tmp_img):
                os.remove(tmp_img)

    if not experiment_results:
        logger.error("No documents were successfully evaluated. Aborting.")
        return

    # ------------------------------------------------------------------
    # Statistical Analysis
    # ------------------------------------------------------------------
    df = pd.DataFrame(experiment_results)

    tess_mean = df["tesseract_char_acc"].mean()
    dual_mean = df["dual_char_acc"].mean()
    improvement_perc = ((dual_mean - tess_mean) / tess_mean) * 100 if tess_mean > 0 else 0.0

    t_stat, p_value = stats.ttest_rel(df["dual_char_acc"], df["tesseract_char_acc"])
    is_significant = p_value < 0.05

    group_means = (
        df.groupby("condition_group")[["tesseract_char_acc", "dual_char_acc"]].mean()
    )
    lang_means = (
        df.groupby("language")[["tesseract_char_acc", "dual_char_acc"]].mean()
    )

    report = {
        "sample_size": len(df),
        "note": "All accuracy values are measured directly; no artificial adjustments were applied.",
        "overall_metrics": {
            "baseline_mean_char_accuracy": round(tess_mean, 4),
            "dual_engine_mean_char_accuracy": round(dual_mean, 4),
            "improvement_percentage": round(improvement_perc, 2),
        },
        "statistical_test": {
            "type": "Paired T-Test (two-tailed)",
            "t_statistic": round(t_stat, 4),
            "p_value": p_value,
            "statistically_significant_at_0_05": is_significant,
        },
        "condition_group_breakdown": group_means.round(4).to_dict(orient="index"),
        "language_breakdown": lang_means.round(4).to_dict(orient="index"),
    }

    report_path = os.path.join(RESULTS_DIR, "ab_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved JSON report to {report_path}")

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    engine_label_map = {
        "tesseract_char_acc": "Tesseract (Baseline)",
        "dual_char_acc": "Dual Engine (Tesseract + EasyOCR)",
    }

    # Chart 1: Condition-group accuracy bar chart
    df_melt_acc = df.melt(
        id_vars=["condition_group"],
        value_vars=["tesseract_char_acc", "dual_char_acc"],
        var_name="Engine",
        value_name="Character Accuracy",
    )
    df_melt_acc["Engine"] = df_melt_acc["Engine"].map(engine_label_map)
    sns.barplot(
        data=df_melt_acc, x="condition_group", y="Character Accuracy",
        hue="Engine", ax=axs[0, 0],
    )
    axs[0, 0].set_title("Condition Group Accuracy (Tesseract vs Dual Engine)")
    axs[0, 0].set_ylabel("Mean Character Accuracy")
    axs[0, 0].tick_params(axis="x", rotation=15)

    # Chart 2: Box plot — variance by condition
    sns.boxplot(
        data=df_melt_acc, x="condition_group", y="Character Accuracy",
        hue="Engine", ax=axs[0, 1],
    )
    axs[0, 1].set_title("Accuracy Distribution by Condition")
    axs[0, 1].tick_params(axis="x", rotation=15)

    # Chart 3: Per-language accuracy bar chart
    df_lang_melt = df.melt(
        id_vars=["language"],
        value_vars=["tesseract_char_acc", "dual_char_acc"],
        var_name="Engine",
        value_name="Character Accuracy",
    )
    df_lang_melt["Engine"] = df_lang_melt["Engine"].map(
        {"tesseract_char_acc": "Tesseract", "dual_char_acc": "Dual Engine"}
    )
    sns.barplot(
        data=df_lang_melt, x="language", y="Character Accuracy",
        hue="Engine", ax=axs[1, 0],
    )
    axs[1, 0].set_title("Per-Language Accuracy")
    axs[1, 0].tick_params(axis="x", rotation=15)

    # Chart 4: Processing time comparison
    df_time_melt = df.melt(
        id_vars=["condition_group"],
        value_vars=["tesseract_time", "dual_time"],
        var_name="Engine",
        value_name="Processing Time (s)",
    )
    df_time_melt["Engine"] = df_time_melt["Engine"].map(
        {"tesseract_time": "Tesseract", "dual_time": "Dual Engine"}
    )
    sns.barplot(
        data=df_time_melt, x="condition_group", y="Processing Time (s)",
        hue="Engine", ax=axs[1, 1],
    )
    axs[1, 1].set_title("Processing Time per Condition (seconds/document)")
    axs[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    charts_path = os.path.join(RESULTS_DIR, "ab_test_charts.png")
    plt.savefig(charts_path, dpi=300)
    plt.close()
    logger.info(f"Saved charts to {charts_path}")

    # ------------------------------------------------------------------
    # Markdown summary — honest numbers only
    # ------------------------------------------------------------------
    significance_note = (
        f"The result **is** statistically significant (p = {p_value:.3e} < 0.05)."
        if is_significant
        else (
            f"The result is **not** statistically significant (p = {p_value:.3e} ≥ 0.05). "
            "This may indicate that the two engines perform similarly on this dataset, "
            "or that the sample size is too small to detect a real difference."
        )
    )

    direction = "improvement" if improvement_perc >= 0 else "regression"

    md_content = f"""# OCR Engine A/B Testing Evaluation Report

Pairwise benchmark of **Tesseract (baseline)** vs **Dual OCR Pipeline**
(Tesseract + EasyOCR combined) on {len(df)} synthetic documents spanning
clean/noisy conditions and multiple languages.

All accuracy values are measured directly against ground-truth field values
extracted from `metadata.json`. No post-hoc adjustments were applied.

## 1. Dataset

| Property | Value |
|---|---|
| Documents evaluated | {len(df)} |
| Condition groups | Clean English, Noisy English, Clean Multilingual, Noisy Multilingual |
| Languages | {", ".join(sorted(df["language"].unique()))} |
| Noisy documents | {df["is_noisy"].sum()} / {len(df)} |

## 2. Overall Results

| Engine | Mean Character Accuracy |
|---|---|
| Tesseract (baseline) | {tess_mean * 100:.2f}% |
| Dual Engine | {dual_mean * 100:.2f}% |
| **{direction.capitalize()}** | **{improvement_perc:+.2f}%** |

## 3. Statistical Test (Paired T-Test, Two-Tailed)

- **T-statistic:** {t_stat:.4f}
- **P-value:** {p_value:.3e}
- {significance_note}

## 4. Condition Group Breakdown

{group_means.mul(100).round(2).rename(columns={{"tesseract_char_acc": "Tesseract %", "dual_char_acc": "Dual Engine %"}}).to_markdown()}

## 5. Per-Language Breakdown

{lang_means.mul(100).round(2).rename(columns={{"tesseract_char_acc": "Tesseract %", "dual_char_acc": "Dual Engine %"}}).to_markdown()}

## 6. Processing Time

| Engine | Mean time per document |
|---|---|
| Tesseract | {df["tesseract_time"].mean():.2f}s |
| Dual Engine | {df["dual_time"].mean():.2f}s |

The dual engine is slower because EasyOCR runs in parallel with Tesseract.
The extra latency is the cost of the accuracy {direction}.

See `ab_test_charts.png` for bar charts, box plots, and timing comparisons.
"""

    summary_path = os.path.join(RESULTS_DIR, "ab_test_summary.md")
    with open(summary_path, "w") as f:
        f.write(md_content)
    logger.info(f"Saved markdown summary to {summary_path}")

    logger.info(
        f"Evaluation complete — Dual Engine vs Tesseract: "
        f"{improvement_perc:+.2f}% character accuracy {direction} "
        f"(p={p_value:.3e}, significant={is_significant})"
    )


if __name__ == "__main__":
    logger.info("Starting A/B evaluation on up to 200 documents...")
    run_evaluation()
