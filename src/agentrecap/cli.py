import argparse
import webbrowser
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .adapters import ADAPTERS, add_input_arguments, inputs_from_args
from .report import build_report, run_pipeline


def main() -> None:
    default_output_dir = (
        Path.home()
        / ".agentrecap"
        / "reports"
        / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser = argparse.ArgumentParser(
        description="Analyze local coding-agent sessions and create an offline HTML report."
    )
    add_input_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Report directory (default: ~/.agentrecap/reports/<timestamp>)",
    )
    parser.add_argument("--title", default="agentrecap report")
    parser.add_argument(
        "--since",
        dest="since_date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Analyze events on or after this local date",
    )
    parser.add_argument(
        "--until",
        dest="until_date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Analyze events on or before this local date",
    )
    parser.add_argument("--open", action="store_true", help="Open the finished report in the default browser")
    args = parser.parse_args()

    if args.since_date and args.until_date and args.since_date > args.until_date:
        parser.error("--since must be on or before --until")

    inputs = {
        source: path.expanduser().resolve()
        for source, path in inputs_from_args(args).items()
    }

    print("Analysing...")

    output_dir = args.output_dir.expanduser().resolve()
    if not any(ADAPTERS[source].discover_sessions(path) for source, path in inputs.items()):
        parser.error("No coding agent sessions found on this machine")

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = (
        datetime.combine(args.since_date, time.min).astimezone()
        if args.since_date
        else None
    )
    end_time = (
        datetime.combine(args.until_date + timedelta(days=1), time.min).astimezone()
        if args.until_date
        else None
    )
    try:
        run_pipeline(inputs, output_dir, start_time=start_time, end_time=end_time)
    except ValueError as error:
        parser.error(str(error))

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
