#!/bin/bash
# Development helper script for PPoGA project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 PPoGA Development Helper${NC}"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv .venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install/update dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run different commands based on argument
case "$1" in
    "test")
        echo -e "${GREEN}Running all tests...${NC}"
        pytest tests/ -v
        ;;
    "test-unit")
        echo -e "${GREEN}Running unit tests...${NC}"
        pytest tests/unit/ -v
        ;;
    "test-integration")
        echo -e "${GREEN}Running integration tests...${NC}"
        pytest tests/integration/ -v
        ;;
    "test-benchmark")
        echo -e "${GREEN}Running benchmark tests...${NC}"
        pytest tests/benchmark/ -v --benchmark-only
        ;;
    "lint")
        echo -e "${GREEN}Running linting...${NC}"
        flake8 src/
        black --check src/
        isort --check-only src/
        mypy src/ --ignore-missing-imports
        ;;
    "format")
        echo -e "${GREEN}Formatting code...${NC}"
        black src/
        isort src/
        ;;
    "coverage")
        echo -e "${GREEN}Running tests with coverage...${NC}"
        pytest tests/ --cov=src/ppoga_project --cov-report=html --cov-report=term
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    "clean")
        echo -e "${GREEN}Cleaning up...${NC}"
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        rm -rf htmlcov/ .coverage .pytest_cache/ .mypy_cache/
        ;;
    "install")
        echo -e "${GREEN}Setting up development environment...${NC}"
        pip install -e .
        ;;
    *)
        echo -e "${YELLOW}Usage: $0 {test|test-unit|test-integration|test-benchmark|lint|format|coverage|clean|install}${NC}"
        echo ""
        echo "Commands:"
        echo "  test            - Run all tests"
        echo "  test-unit       - Run unit tests only"
        echo "  test-integration - Run integration tests only"
        echo "  test-benchmark  - Run benchmark tests only"
        echo "  lint            - Run linting checks"
        echo "  format          - Format code with black and isort"
        echo "  coverage        - Run tests with coverage report"
        echo "  clean           - Clean up cache files"
        echo "  install         - Install package in development mode"
        ;;
esac

echo -e "${GREEN}✅ Done!${NC}"
