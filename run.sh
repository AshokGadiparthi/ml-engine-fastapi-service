#!/bin/bash
# Run ML Engine FastAPI Service

# Set environment variables
export ML_ENGINE_PATH="/home/claude/kedro-ml-engine/src"
export PYTHONPATH="${ML_ENGINE_PATH}:${PYTHONPATH}"

# Change to app directory
cd /home/claude/ml-engine-api

# Install dependencies if needed
# pip install -r requirements.txt

# Run the server
echo "Starting ML Engine API on http://localhost:8000"
echo "Docs available at http://localhost:8000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
