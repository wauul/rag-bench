FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 RAGAS_DO_NOT_TRACK=true DATA_DIR=/app/data
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY sample_data sample_data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
