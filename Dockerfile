# Using official Python image
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/fast_api
RUN mkdir -p /app/models

# The actual application will be mounted via volumes
VOLUME /app/fast_api
VOLUME /app/models

# Set the working directory to where main.py lives
WORKDIR /app/fast_api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
