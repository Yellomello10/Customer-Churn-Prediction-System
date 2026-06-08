# Use lightweight official Python runtime as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy dependency requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, models, clean data, and frontend assets
COPY src/ ./src/
COPY data/ ./data/
COPY static/ ./static/
COPY outputs/ ./outputs/
COPY app.py .
COPY main.py .

# Expose port the app runs on
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
