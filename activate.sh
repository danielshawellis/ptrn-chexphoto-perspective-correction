#!/bin/bash

# Define your virtual environment directory
VENV_DIR=".venv"

# Step 1: Create a new virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3.8 -m venv $VENV_DIR
    echo "Created virtual environment in $VENV_DIR"
fi

# Step 2: Activate the virtual environment
source $VENV_DIR/bin/activate
echo "Activated virtual environment"
