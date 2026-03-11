# Multi-stage Dockerfile for FastAPI Backend
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies (minimal for build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel for better performance
# Suppress PATH warnings by adding to PATH early
ENV PATH=/root/.local/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Strategy: Install PyTorch CPU-only first (largest package, ~500MB)
# Then install rest from requirements.txt - pip will skip already installed packages
RUN pip install --no-cache-dir --user \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install packaging first (required by langchain-core)
RUN pip install --no-cache-dir --user packaging

# Install all other dependencies from requirements.txt
# Pip will skip torch/torchvision since they're already installed
RUN pip install --no-cache-dir --user -r requirements.txt

# Clean up pip cache and temporary files to reduce image size
RUN pip cache purge || true
RUN rm -rf /tmp/* /var/tmp/* /root/.cache/pip || true

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder (optimized - only site-packages and bin)
# Exclude __pycache__ and .pyc files to reduce size
COPY --from=builder /root/.local/lib/python3.11/site-packages /root/.local/lib/python3.11/site-packages
COPY --from=builder /root/.local/bin /root/.local/bin

# Clean up Python cache files in copied packages
RUN find /root/.local/lib/python3.11/site-packages -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
RUN find /root/.local/lib/python3.11/site-packages -type f -name "*.pyc" -delete 2>/dev/null || true
RUN find /root/.local/lib/python3.11/site-packages -type f -name "*.pyo" -delete 2>/dev/null || true

# Make sure scripts in .local are usable (fixes PATH warnings)
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.11/site-packages:$PYTHONPATH

# Copy only necessary application code (excludes files in .dockerignore)
COPY . .

# Create necessary directories
RUN mkdir -p reports data/rag_index

# Remove Python cache and temporary files from application code
RUN find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
RUN find . -type f -name "*.pyc" -delete 2>/dev/null || true
RUN find . -type f -name "*.pyo" -delete 2>/dev/null || true
RUN find . -type f -name "*.pyd" -delete 2>/dev/null || true

# Verify essential model files exist (optional check)
RUN python -c "import os; \
    assert os.path.exists('app/ml_models/xray/pneumonia_resnet50.pth'), 'X-ray model missing'; \
    print('✓ Essential model files verified')" || echo "⚠️  Warning: Some model files may be missing"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway sets PORT dynamically)
EXPOSE ${PORT:-8000}

# Health check (use PORT env var or default to 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# Run the application (Railway sets PORT automatically)
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"

