# __init__.py in the root directory

import sys
import os

# Get the current directory of this script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add both the current directory and its parent to sys.path
sys.path.append(current_dir)

