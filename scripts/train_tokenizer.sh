#!/bin/bash

# Ensure script stops on error
set -e

# Run tokenizer training
echo "Training tokenizer..."
python -m src.tokenizer "$@"
