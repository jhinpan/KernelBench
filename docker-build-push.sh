#!/bin/bash
#
# Build and push the KernelBench image to Docker Hub.
# The script works with either Docker Compose v1 (“docker-compose”)
# or v2 (“docker compose”) – it auto-detects which command is present.

# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
DOCKER_USERNAME="jhinpan"
IMAGE_NAME="kernelbenchimage"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info()    { printf "${GREEN}[INFO]${NC} %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
error()   { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; exit 1; }

# ────────────────────────────────────────────────────────────────────
# Detect compose command (v1 vs. v2)
# ────────────────────────────────────────────────────────────────────
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    error "Docker Compose not found. Install either docker-compose-plugin (v2) or docker-compose (v1)."
fi
info "Using Compose command: ${COMPOSE_CMD}"

# ────────────────────────────────────────────────────────────────────
# Parse CLI flags
# ────────────────────────────────────────────────────────────────────
TAG="latest"
PUSH_ONLY=false

case "$1" in
  ""|latest) ;;                                   # default
  --push-only) PUSH_ONLY=true; TAG="${2:-latest}" ;;
  *) TAG="$1" ;;
esac

# ────────────────────────────────────────────────────────────────────
# Login
# ────────────────────────────────────────────────────────────────────
info "Logging in to Docker Hub…"
docker login || error "Docker login failed."

# ────────────────────────────────────────────────────────────────────
# Build
# ────────────────────────────────────────────────────────────────────
if ! $PUSH_ONLY; then
    info "Building image ${FULL_IMAGE_NAME}:${TAG}"
    $COMPOSE_CMD build || error "Build failed."

    # Re-tag if custom tag requested
    if [[ "$TAG" != "latest" ]]; then
        info "Tagging image as ${FULL_IMAGE_NAME}:${TAG}"
        docker tag "${FULL_IMAGE_NAME}:latest" "${FULL_IMAGE_NAME}:${TAG}"
    fi
else
    info "Skipping build (push-only mode)"
fi

# ────────────────────────────────────────────────────────────────────
# Push
# ────────────────────────────────────────────────────────────────────
info "Pushing ${FULL_IMAGE_NAME}:${TAG} to Docker Hub…"
docker push "${FULL_IMAGE_NAME}:${TAG}" || error "Push failed."

# Also push “latest” when a specific tag built this run
if ! $PUSH_ONLY && [[ "$TAG" != "latest" ]]; then
    info "Pushing ${FULL_IMAGE_NAME}:latest"
    docker push "${FULL_IMAGE_NAME}:latest" || warn "Could not push latest tag."
fi

# ────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────
info "Image available at: https://hub.docker.com/r/${FULL_IMAGE_NAME}"
info "Local image size:"
docker images "${FULL_IMAGE_NAME}:${TAG}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"