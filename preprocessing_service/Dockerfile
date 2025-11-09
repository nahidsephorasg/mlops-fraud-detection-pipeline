FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .

EXPOSE 8031


HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8031/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8031"]
