#!/bin/bash
# Shell script to activate conda environment and run dialogue system
# Usage: ./run_dialogue.sh

ENV_NAME="5112Project"

echo "Activating conda environment: $ENV_NAME..."

# Try to activate conda environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$ENV_NAME'"
    echo "Please make sure conda is installed and the environment exists"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Environment activated successfully!"
echo "Running dialogue system..."
echo "=================================================="

# Run the Python script
python dialogue_system.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo
    echo "Script execution failed. Press Enter to exit..."
    read
fi