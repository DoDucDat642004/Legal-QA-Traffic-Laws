#!/bin/bash

# Start the FastAPI backend in the background
echo "Starting FastAPI backend on port 8002..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8002 &

# Wait a moment for the backend to initialize
sleep 5

# Start the Streamlit frontend
echo "Starting Streamlit frontend on port 7860..."
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
