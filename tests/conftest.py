import sys
from pathlib import Path

# The project code lives in scripts/, which is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
