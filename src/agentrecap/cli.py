"""Command-line entry point for agentrecap."""

import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

from .report import build_report, discover_jsonl, run_pipeline


def main() -> None:
    default_output_dir = (
        Path.home()
        / ".agentrecap"
        / "reports"
        / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser = argparse.ArgumentParser(
        description="Convert Codex and Claude sessions, analyze them, and create an offline HTML report."
    )
    parser.add_argument("--codex-input", type=Path, default=Path.home() / ".codex" / "sessions")
    parser.add_argument("--claude-input", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Report directory (default: ~/.agentrecap/reports/<timestamp>)",
    )
    parser.add_argument("--title", default="Codex and Claude usage report")
    parser.add_argument("--open", action="store_true", help="Open the finished report in the default browser")
    args = parser.parse_args()

    codex_input = args.codex_input.expanduser().resolve()
    claude_input = args.claude_input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not discover_jsonl(codex_input) and not discover_jsonl(claude_input):
        parser.error(f"No JSONL transcripts found in {codex_input} or {claude_input}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(codex_input, claude_input, output_dir)
    index_path = build_report(output_dir, args.title)
    print(f'Generated report at "{index_path}"')
    if args.open:
        webbrowser.open(index_path.as_uri())
        return

    try:
        should_open = input("Would you like to open it in the browser? (y/n) ")
    except EOFError:
        return
    if should_open.strip().lower() in {"y", "yes"}:
        webbrowser.open(index_path.as_uri())


if __name__ == "__main__":
    main()
