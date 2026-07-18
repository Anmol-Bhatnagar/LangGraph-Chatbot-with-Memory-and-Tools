import sys
import os

# Add the 'src' directory to the path so pytest can find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
