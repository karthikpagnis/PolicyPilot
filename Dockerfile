# Dockerfile for PolicyPilot FastAPI app on Render
# Installs Python, Tesseract, and Ollama

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        curl \
        git \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && rm -rf /var/lib/apt/lists/*

# Install Ollama (Linux x86_64)
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port for Uvicorn
EXPOSE 8000

# Start Ollama in the background and then run FastAPI
CMD ollama serve & uvicorn main:app --host 0.0.0.0 --port 8000
