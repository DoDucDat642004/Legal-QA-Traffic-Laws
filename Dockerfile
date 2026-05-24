# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure entrypoint script is executable
RUN chmod +x entrypoint.sh

# Expose ports (HF Spaces uses 7860 by default for the main UI)
EXPOSE 7860
EXPOSE 8002

# Set environment variables for the application
# These can be overridden in HF Spaces settings
ENV RAG_VECTOR_BACKEND=local
ENV RAG_GRAPH_BACKEND=local
ENV RAG_CANONICAL_BACKEND=local
ENV RAG_OBJECT_BACKEND=local

# Run the startup script
ENTRYPOINT ["./entrypoint.sh"]
