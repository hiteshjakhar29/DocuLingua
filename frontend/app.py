import streamlit as st
import requests
import pandas as pd
import time
import os
import random

API_URL = os.getenv("API_URL", "http://localhost:8000")
_API_KEY = os.getenv("API_KEY", "9XnaSwenx8sN-384Nnrqkb1wb0gzpc6V5vdYjXfluL4")
_AUTH_HEADERS = {"X-API-Key": _API_KEY}

st.set_page_config(page_title="DocuLingua - Multilingual Document Intelligence", layout="wide", page_icon="📄")

# Define Document Icons
DOC_ICONS = {
    "university_degree": "🎓",
    "transcript": "📝",
    "professional_license": "⚕️",
    "employment_letter": "🏢",
    "diploma": "📜",
    "certificate": "🏅",
    "unknown": "❓"
}

def get_color(confidence):
    """Returns CSS color based on 0-100 confidence scale."""
    if confidence >= 85:
        return "green"
    elif confidence >= 60:
        return "orange"
    return "red"

def fetch_stats():
    """Fetches real-time pipeline statistics."""
    try:
        response = requests.get(f"{API_URL}/stats", headers=_AUTH_HEADERS, timeout=5)
        if response.status_code == 200:
            return response.json().get("statistics", {})
    except requests.exceptions.RequestException:
        return None

def process_document(file_bytes, filename):
    """Executes the 3-step FastAPI Document Pipeline securely with Progress Hook."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Phase 1: Uploading
        status_text.text("Phase 1/5: Uploading Document...")
        progress_bar.progress(10)
        
        files = {"file": (filename, file_bytes)}
        res_upload = requests.post(f"{API_URL}/upload", files=files, headers=_AUTH_HEADERS)
        if res_upload.status_code == 401:
            st.error("Authentication failed: invalid API key. Check the API_KEY in your .env.")
            return None
        if res_upload.status_code == 403:
            st.error("Authentication failed: X-API-Key header missing. Set API_KEY in your .env.")
            return None
        if res_upload.status_code != 200:
            st.error(f"Upload failed ({res_upload.status_code}): {res_upload.text}")
            return None

        doc_id = res_upload.json().get("document_id")

        # Phase 2: Dual OCR Execution
        status_text.text("Phase 2/5: Running Dual OCR Engines (Tesseract & EasyOCR)...")
        progress_bar.progress(30)

        # Phase 3: Classification & Pipeline Invocation
        status_text.text("Phase 3/5: Classifying & Extracting Entities...")
        progress_bar.progress(50)

        res_analyze = requests.post(f"{API_URL}/analyze/{doc_id}", headers=_AUTH_HEADERS)
        if res_analyze.status_code != 200:
            st.error(f"Analysis failed ({res_analyze.status_code}): {res_analyze.text}")
            return None

        # Phase 4: Formatting Score Array
        status_text.text("Phase 4/5: Calculating Field Confidence Variables...")
        progress_bar.progress(80)
        time.sleep(0.5)

        # Phase 5: Fetching final compiled structure
        status_text.text("Phase 5/5: Rendering Client Matrices...")
        progress_bar.progress(100)

        res_results = requests.get(f"{API_URL}/results/{doc_id}", headers=_AUTH_HEADERS)
        status_text.empty()
        progress_bar.empty()

        if res_results.status_code == 200:
            return res_results.json()
        return None

    except requests.exceptions.ConnectionError:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Connection error: cannot reach the API at {API_URL}. Is it running?")
        return None
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Pipeline Execution Error: {e}")
        return None

# ==========================================
# Sidebar UI
# ==========================================
st.sidebar.title("DocuLingua Setup")

st.sidebar.markdown("### 📊 Live Statistics")
stats = fetch_stats()
if stats:
    st.sidebar.metric("Total Processing Volume", stats.get("total_documents_analyzed", 0))
    st.sidebar.metric("Average API Latency", f"{stats.get('pipeline_average_time_seconds', 0.0)}s")
    st.sidebar.metric("System Global Confidence", f"{stats.get('system_average_confidence_percentage', 0.0)}%")
else:
    st.sidebar.warning("API Unreachable - Stats offline.")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🚀 Execution Tools")
use_example = st.sidebar.button("Use Simulated Example")

uploaded_file = st.sidebar.file_uploader("Or Upload Custom Document", type=["pdf", "png", "jpg", "jpeg"])

# ==========================================
# Main Canvas UI
# ==========================================
st.title("DocuLingua - Multilingual Document Intelligence")
st.markdown("*A highly distributed extraction engine applying automated Confidence Scoring across nested OCR & NER models.*")

file_to_process = None
filename = "upload.pdf"

# Handle Examlpe Setup
if use_example:
    synth_dir = "data/synthetic_docs"
    if os.path.exists(synth_dir):
        types = [d for d in os.listdir(synth_dir) if os.path.isdir(os.path.join(synth_dir, d))]
        if types:
            random_type = random.choice(types)
            type_dir = os.path.join(synth_dir, random_type)
            pdfs = [f for f in os.listdir(type_dir) if f.endswith('.pdf')]
            if pdfs:
                random_pdf = random.choice(pdfs)
                filepath = os.path.join(type_dir, random_pdf)
                with open(filepath, "rb") as f:
                    file_to_process = f.read()
                filename = random_pdf
                st.session_state["file_bytes"] = file_to_process
                st.session_state["filename"] = filename
                st.success(f"Loaded Demo Document: `{random_pdf}` ({random_type})")
    else:
        st.sidebar.error("Synthetic documents directory missing. Generate them first.")

if uploaded_file:
    file_to_process = uploaded_file.getvalue()
    filename = uploaded_file.name
    st.session_state["file_bytes"] = file_to_process
    st.session_state["filename"] = filename

# Persist loaded file through streamlit reruns
if "file_bytes" in st.session_state:
    file_to_process = st.session_state["file_bytes"]
    filename = st.session_state["filename"]

if file_to_process:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Document Preview")
        if filename.lower().endswith(".pdf"):
            st.markdown(f"*(Previewing PDF Metadata for {filename})*")
        else:
            st.image(file_to_process, use_container_width=True)
            
        analyze_btn = st.button("Extract Data ➡️", type="primary", use_container_width=True)
        
    with col2:
        if analyze_btn:
            results = process_document(file_to_process, filename)
            
            if results:
                st.subheader("Extraction Results")
                
                # Confidence Alert
                g_conf = results.get("global_confidence", 0.0)
                is_manual = results.get("manual_review_required", False)
                
                if is_manual:
                    st.error(f"⚠️ MANUAL REVIEW REQUIRED — Global Confidence dropped to **{g_conf}%** spanning **{results.get('flagged_fields', 0)}** flagged field(s).")
                else:
                    st.success(f"✅ HIGH CONFIDENCE — System automatically cleared document at **{g_conf}%**.")
                
                # Expanders
                with st.expander("📄 Document Classification", expanded=True):
                    doc_type = results.get('document_type', 'unknown')
                    icon = DOC_ICONS.get(doc_type, "❓")
                    st.markdown(f"### {icon} `{doc_type.upper()}`")
                    st.caption("Auto-discovered via `bert-base-multilingual-cased` Pipeline")
                    
                with st.expander("📊 Extracted Entity Breakdown", expanded=True):
                    fields = results.get("extracted_fields", {})
                    if fields:
                        # Compile nice DF
                        df_data = []
                        for label, matches in fields.items():
                            for match in matches:
                                df_data.append({
                                    "Label": label,
                                    "Value": match["value"],
                                    "Score": match["confidence_score"]
                                })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.dataframe(
                                df.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100),
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        st.info("No Entities safely extracted matching 70%+ probabilities.")

                with st.expander("🛠️ Low Confidence Traps", expanded=is_manual):
                    trap_exists = False
                    for label, matches in fields.items():
                        for match in matches:
                            if match.get("requires_review"):
                                trap_exists = True
                                st.warning(f"**{label}** -> `{match['value']}` (Scored inherently low at **{match['confidence_score']}%** matrix)")
                    if not trap_exists:
                        st.info("Zero fields violated the explicit threshold rules.")
                        
                with st.expander("⚙️ Raw API Output", expanded=False):
                    st.json(results)
                    
                # Footer formatting
                st.markdown("---")
                st.caption(f"Engine Latency: `{results.get('processing_time_seconds', 0.0)}s` | Tracing ID: `{results.get('document_id')}`")
else:
    # If no file is actively tracked
    st.info("👈 Upload a graphic document or use an explicit simulated sample from the configuration bar to initiate the complete Unified Inference schema.")
