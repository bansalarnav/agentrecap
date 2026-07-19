"""Command-line entry point for agentrecap."""

import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

from .adapters import ADAPTERS
from .report import build_report, run_pipeline


def main() -> None:
    default_output_dir = (
        Path.home()
        / ".agentrecap"
        / "reports"
        / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser = argparse.ArgumentParser(
        description="Convert Codex, Claude, and Cursor sessions, analyze them, and create an offline HTML report."
    )
    for source, adapter in ADAPTERS.items():
        parser.add_argument(
            f"--{source}-input",
            type=Path,
            default=adapter.DEFAULT_INPUT,
            help=adapter.INPUT_HELP,
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Report directory (default: ~/.agentrecap/reports/<timestamp>)",
    )
    parser.add_argument("--title", default="Coding agent usage report")
    parser.add_argument("--open", action="store_true", help="Open the finished report in the default browser")
    args = parser.parse_args()

    inputs = {
        source: getattr(args, f"{source}_input").expanduser().resolve()
        for source in ADAPTERS
    }
    output_dir = args.output_dir.expanduser().resolve()
    if not any(ADAPTERS[source].discover_sessions(path) for source, path in inputs.items()):
        paths = " or ".join(str(path) for path in inputs.values())
        parser.error(f"No session data found in {paths}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Analysing...")
    run_pipeline(inputs, output_dir)
    index_path = build_report(output_dir, args.title)
    print(f'Generated report at "{index_path}"')
    if args.open:
        webbrowser.open(index_path.as_uri())
        return

    try:
        should_open = input("Would you like to open it in the browser? (y/n) ")
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if should_open.strip().lower() in {"y", "yes"}:
        webbrowser.open(index_path.as_uri())


if __name__ == "__main__":
    main()
