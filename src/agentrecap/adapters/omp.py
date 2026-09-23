"""Oh My Pi sessions use Pi's JSONL format under a separate agent directory."""

from pathlib import Path

from . import pi

SOURCE = "omp"
PROVIDER = "omp"
DISPLAY_NAME = "Oh My Pi"
GRAPH_COLOR = "tab:olive"
DEFAULT_INPUT = Path.home() / ".omp" / "agent"
INPUT_HELP = "Oh My Pi agent directory containing sessions/"

discover_sessions = pi.discover_sessions
finalize_events = pi.finalize_events


def convert_thread(path: Path) -> list[dict]:
    return pi.convert_thread(path, source=SOURCE)
