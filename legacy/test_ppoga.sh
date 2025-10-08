#!/bin/bash
# PPoGA-on-PoG Test Runner
# Quick test script for the PPoGA system

echo "🚀 PPoGA-on-PoG Test Runner"
echo "============================"

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install poetry first."
    echo "   Visit: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install dependencies if not already installed
echo "📦 Installing dependencies..."
poetry install

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "   Copy .env.example to .env and configure your Azure OpenAI credentials"
    echo "   For testing, you can use mock mode:"
    echo "   poetry run python -m ppoga_project.main_ppoga_on_pog \"Who directed The Godfather?\" --mock"
    echo ""
fi

# Test questions
questions=(
    "Who directed The Godfather?"
    "Who is the spouse of the director of The Godfather?"
    "What movies did Steven Spielberg direct?"
    "When was Barack Obama born?"
    "What is the capital of France?"
)

echo ""
echo "🧪 Test Questions Available:"
for i in "${!questions[@]}"; do
    echo "   $((i+1)). ${questions[$i]}"
done

echo ""
echo "Usage Examples:"
echo "==============="
echo ""
echo "1. Run with real Azure OpenAI API:"
echo "   poetry run python -m ppoga_project.main_ppoga_on_pog \"Who directed The Godfather?\""
echo ""
echo "2. Run in mock mode (no Azure API needed):"
echo "   poetry run python -m ppoga_project.main_ppoga_on_pog \"Who directed The Godfather?\" --mock"
echo ""
echo "3. Quick test with mock mode:"
read -p "Press Enter to run quick mock test, or Ctrl+C to exit..."

echo ""
echo "🧪 Running quick mock test..."
poetry run python -m ppoga_project.main_ppoga_on_pog "${questions[0]}" --mock
