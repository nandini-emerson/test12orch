

FROM python:3.11-slim

# Install OS dependencies for cryptography, pandas, httpx, uvicorn, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        gcc \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Create app directory
WORKDIR /app

# Copy requirements file if present, else install directly
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set default environment variables for Azure/OpenAI if needed
# (These should be set at runtime via secrets/environment)
ENV PORT=8000

# Startup command
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

