#!/bin/bash

# Step 1: Run the activate.sh script to create and activate the virtual environment
source ./activate.sh

# Step 2: Remove all existing packages
INSTALLED_PACKAGES=$(pip freeze)
if [ -n "$INSTALLED_PACKAGES" ]; then
    echo "$INSTALLED_PACKAGES" | xargs pip uninstall -y
    echo "Removed all existing packages"
else
    echo "No packages installed"
fi

# Step 3: Install packages from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Installed packages from requirements.txt"
else
    echo "requirements.txt file not found"
fi

# Step 4: Install IPython kernel in the virtual environment
pip install ipykernel
echo "Installed IPython kernel"

# Step 5: Add the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
echo "Added virtual environment as Jupyter kernel 'Python (.venv)'"

# Step 6: Verify installation
echo "Current installed packages:"
pip freeze
