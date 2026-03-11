#!/bin/bash
# Build and push Docker image to Docker Hub
# Usage: ./build-and-push.sh [your-dockerhub-username]

set -e

DOCKER_USERNAME=${1:-"your-username"}
IMAGE_NAME="doctor-assistant-backend"
TAG="latest"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "🔨 Building Docker image..."
docker build -t ${FULL_IMAGE_NAME} -f Dockerfile .

echo "📤 Pushing to Docker Hub..."
docker push ${FULL_IMAGE_NAME}

echo "✅ Image pushed: ${FULL_IMAGE_NAME}"
echo ""
echo "Now in Railway:"
echo "1. Go to Settings → Source"
echo "2. Change from 'Dockerfile' to 'Docker Image'"
echo "3. Enter: ${FULL_IMAGE_NAME}"
