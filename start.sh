#!/bin/bash
# Knight AI SDR — Production Start Script
cd "$(dirname "$0")"
echo "Starting Knight AI SDR on port ${PORT:-8080}..."
python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
