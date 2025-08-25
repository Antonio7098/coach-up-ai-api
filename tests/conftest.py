import sys
import os
from pathlib import Path

# Ensure project root is on sys.path for `import app.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default: keep classifier provider disabled unless a test opts in
os.environ.setdefault("AI_CLASSIFIER_ENABLED", "0")
