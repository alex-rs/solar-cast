FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY app ./app
COPY config ./config
ENV DATA_DIR=/app/data
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import os,urllib.request;req=urllib.request.Request('http://127.0.0.1:8000/api/v1/health',headers={'Authorization':'Bearer '+os.environ.get('API_TOKEN','')});urllib.request.urlopen(req)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
