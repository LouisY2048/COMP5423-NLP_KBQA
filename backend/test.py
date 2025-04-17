import os
import sys

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")
backend_dir = os.path.dirname(os.path.dirname(current_dir))
print(f"Backend directory: {backend_dir}")
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Project root: {project_root}")