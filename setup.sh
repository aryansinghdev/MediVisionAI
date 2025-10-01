#!/bin/bash

# Setup script for MediVisionAI environment
echo "==================================================="
echo "Setting up MediVisionAI environment..."
echo "==================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "==================================================="
echo "Environment setup completed successfully!"
echo "==================================================="
echo "To activate the environment, run: source venv/bin/activate"
