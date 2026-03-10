#!/usr/bin/env bash
# run.sh — One-command local dev and build runner
# Usage:
#   ./deploy/run.sh dev       # Start development server (Next.js hot reload)
#   ./deploy/run.sh build     # Production build
#   ./deploy/run.sh docker    # Build and run production Docker container
#   ./deploy/run.sh demo      # Start the FastAPI ML demo server (fraud detection)
#   ./deploy/run.sh qa        # Run Python QA script
#   ./deploy/run.sh clean     # Remove build artifacts

set -euo pipefail

COMMAND="${1:-dev}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo "[run.sh] $*"; }
die() { echo "[run.sh] ERROR: $*" >&2; exit 1; }

case "$COMMAND" in

  dev)
    log "Starting Next.js development server..."
    cd "$PROJECT_ROOT"
    [ -f package.json ] || die "package.json not found. Run: npm init or use the Next.js starter."
    npm run dev
    ;;

  build)
    log "Building Next.js production bundle..."
    cd "$PROJECT_ROOT"
    npm ci
    npm run build
    log "Build complete. Output in .next/"
    ;;

  docker)
    log "Building Docker image and running container..."
    cd "$PROJECT_ROOT"
    docker build -t komalshahid/portfolio:local -f deploy/Dockerfile .
    docker run --rm -p 3000:3000 --name portfolio-local komalshahid/portfolio:local
    ;;

  demo)
    log "Starting FastAPI ML demo server (fraud detection)..."
    DEMO_DIR="$PROJECT_ROOT/projects/project13-dsc680"
    [ -d "$DEMO_DIR" ] || die "Demo directory not found: $DEMO_DIR"
    cd "$DEMO_DIR"
    if [ ! -d "venv" ]; then
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
    else
      source venv/bin/activate
    fi
    uvicorn deploy.app:app --host 0.0.0.0 --port 8000 --reload
    ;;

  qa)
    log "Running Python QA script..."
    cd "$PROJECT_ROOT"
    python3 -m pip install --quiet requests beautifulsoup4 lxml
    python3 qa/qa_script.py
    ;;

  clean)
    log "Cleaning build artifacts..."
    cd "$PROJECT_ROOT"
    rm -rf .next out node_modules/.cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    log "Clean complete."
    ;;

  *)
    echo "Usage: $0 {dev|build|docker|demo|qa|clean}"
    exit 1
    ;;

esac
