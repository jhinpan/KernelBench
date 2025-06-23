#!/bin/bash

# Script to build and push KernelBench Docker image to Docker Hub

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Docker Hub configuration
DOCKER_USERNAME="jhinpan"
IMAGE_NAME="kernelbenchimage"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Parse command line arguments
TAG="${1:-latest}"
PUSH_ONLY=false

if [ "$1" == "--push-only" ]; then
    PUSH_ONLY=true
    TAG="${2:-latest}"
fi

# Login to Docker Hub
print_status "Logging in to Docker Hub..."
docker login

if [ $? -ne 0 ]; then
    print_error "Docker login failed. Please check your credentials."
    exit 1
fi

# Build the image if not push-only
if [ "$PUSH_ONLY" = false ]; then
    print_status "Building Docker image with tag: ${FULL_IMAGE_NAME}:${TAG}"
    
    # Build with docker-compose for consistency
    docker-compose build
    
    if [ $? -ne 0 ]; then
        print_error "Docker build failed."
        exit 1
    fi
    
    # Tag the image if not using 'latest'
    if [ "$TAG" != "latest" ]; then
        print_status "Tagging image as ${FULL_IMAGE_NAME}:${TAG}"
        docker tag "${FULL_IMAGE_NAME}:latest" "${FULL_IMAGE_NAME}:${TAG}"
    fi
else
    print_status "Skipping build (--push-only flag set)"
fi

# Push the image
print_status "Pushing image to Docker Hub: ${FULL_IMAGE_NAME}:${TAG}"
docker push "${FULL_IMAGE_NAME}:${TAG}"

if [ $? -eq 0 ]; then
    print_status "Successfully pushed ${FULL_IMAGE_NAME}:${TAG} to Docker Hub!"
    print_status "Image URL: https://hub.docker.com/r/${FULL_IMAGE_NAME}"
    
    # Also push latest tag if we built a specific version
    if [ "$TAG" != "latest" ] && [ "$PUSH_ONLY" = false ]; then
        print_status "Also pushing latest tag..."
        docker push "${FULL_IMAGE_NAME}:latest"
    fi
else
    print_error "Failed to push image to Docker Hub."
    exit 1
fi

# Print usage instructions
echo ""
print_status "To pull and run this image:"
echo "  docker pull ${FULL_IMAGE_NAME}:${TAG}"
echo "  docker run --gpus all -it --rm ${FULL_IMAGE_NAME}:${TAG}"

# Print image size
print_status "Image size:"
docker images "${FULL_IMAGE_NAME}:${TAG}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"