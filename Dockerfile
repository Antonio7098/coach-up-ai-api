FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Start command: bind to $PORT if provided by the platform (default 8080)
CMD ["sh", "-c", "gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8080}"]
