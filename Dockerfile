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

# Install Ollama using official install script
RUN curl -fsSL https://ollama.com/install.sh | sh

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
