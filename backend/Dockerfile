FROM python:3.10-slim

WORKDIR /app

# Ensure Tesseract engines, PyMuPDF rendering capabilities (libgl1), and general utilities
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-hin \
    tesseract-ocr-por \
    tesseract-ocr-chi-sim \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies mapped dynamically
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Pre-download EasyOCR models into the image so they don't download at runtime
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, download_enabled=True)"

# Sync application source code
COPY . .

# Copy trained ML models into the image
COPY models/ /app/models/

EXPOSE 8000

# Native deployment invocation
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
