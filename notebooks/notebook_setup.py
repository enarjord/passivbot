import os
import sys

# Change to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)

# Add the project root and src directories to Python's path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

print(f"Working directory set to: {os.getcwd()}")
print(f"Python path: {sys.path}")
