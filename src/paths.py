"""Important paths for the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
STYLE = ROOT / "paper.mplstyle"


def relative_path(path: str | Path) -> str:
    """Return a project path in the user-facing ``./...`` format."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    return f"./{path.relative_to(ROOT)}"
