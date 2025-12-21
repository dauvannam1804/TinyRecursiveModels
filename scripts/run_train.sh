#!/bin/bash

# Ensure script stops on error
set -e

# Create directories if they don't exist
mkdir -p checkpoints

# Run training
echo "Starting training..."
python main.py
