#!/bin/bash

# Install required dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Make sure you're running this script from the project root."
    exit 1
fi

# Install the package in editable mode
echo "Installing the package in editable mode..."
pip install -e .

# Set the current directory as the project root
PROJECT_ROOT=$(pwd)

# Add the project root to the PYTHONPATH for this session
echo "Adding project root to PYTHONPATH for this session..."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "Temporary PYTHONPATH is now set."

# Ask the user to make it permanent in .bashrc
echo "Would you like to add this to your bashrc for future sessions? (y/n)"
read -r answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    # Check if PYTHONPATH already includes the project root
    if ! grep -q "$PROJECT_ROOT" ~/.bashrc; then
        echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"" >> ~/.bashrc
        echo "PYTHONPATH added to ~/.bashrc"
    else
        echo "PYTHONPATH already exists in ~/.bashrc, skipping..."
    fi
else
    echo "Skipping addition to ~/.bashrc."
fi

echo "Setup complete. You can now use the tool."
