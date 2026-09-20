"""Bootstrap for notebooks: puts src/ on sys.path so `import utils`, `import backtest`,
`from paths import DATA, FIGURES, RESULTS` all work. Import this first in every notebook.

Project paths live in src/paths.py — this file deliberately defines none of its own.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
