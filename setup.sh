#!/bin/bash

# Robocar Setup Script
echo "Setting up Robocar AutoDrive environment..."

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install requirements
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the autodrive script, use: source venv/bin/activate && python3 autodrive/main.py"
