# Secundo API (Railway). Build: docker build -t secundo-api .   Run: docker run -p 8000:8000 secundo-api
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt
COPY omr_server.py omr_common.py ./
COPY omr ./omr
COPY runs ./runs
# Railway injects $PORT; MONGODB_URI and ALLOWED_ORIGINS come from service variables (never baked in).
CMD ["sh", "-c", "uvicorn omr_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
