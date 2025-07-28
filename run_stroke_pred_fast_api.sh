#!/bin/bash

# Stop and remove any existing container
docker stop stroke-prediction-api 2>/dev/null
docker rm stroke-prediction-api 2>/dev/null

# Build the Docker image (if not already built)
docker build -t stroke-prediction-api .

# Run the container in detached mode
docker run -d \
  --name stroke-prediction-api \
  -p 8000:8000 \
  --network=host \
  -v "$(pwd)/fast_api:/app/fast_api" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/fast_api/.env:/app/fast_api/.env" \
  -w /app/fast_api \
  thatojoe/stroke-prediction-api

echo "API container running in background"
echo "Access the API at http://localhost:8000"
echo "To view logs: docker logs -f stroke-prediction-api"
