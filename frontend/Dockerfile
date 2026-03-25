FROM python:3.10-slim

WORKDIR /app

# Install isolated requirements alongside Streamlit bindings explicitly
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir streamlit pandas requests -r requirements.txt

# Migrate source context
COPY . .

EXPOSE 8501

# Streamlit Health hook to safely unblock docker-compose wait states
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch production server binding to 0.0.0.0
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
