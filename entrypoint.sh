#!/bin/sh

# Start FastAPI server in the background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Celery worker in the background
touch celery_worker.log
celery -A celery_worker worker --loglevel=info >> celery_worker.log 2>&1 &

# Wait for background jobs
wait