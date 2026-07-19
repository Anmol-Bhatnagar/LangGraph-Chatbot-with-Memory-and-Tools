import sys
import os

# Use a separate test database for pytest to avoid database locking issues with the running server
os.environ["DB_PATH"] = "test_memories.db"

# Add the 'src' directory to the path so pytest can find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
