import argparse
import os
import shutil
import sys
import webbrowser
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .adapters import ADAPTERS, add_input_arguments, inputs_from_args
from .server import DEFAULT_PORT as SERVER_DEFAULT_PORT
from .server import DEFAULT_REFRESH_MINUTES as SERVER_DEFAULT_REFRESH_MINUTES
from .server import refresh_label, serve


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
    parser.add_argument(
        "--server",
        action="store_true",
        help="Keep running and serve the report on localhost, rerunning the analysis periodically",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_DEFAULT_PORT,
        help=f"Port for --server (default: {SERVER_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--refresh-minutes",
        type=float,
        default=SERVER_DEFAULT_REFRESH_MINUTES,
        metavar="MINUTES",
        help=f"How often --server reruns the analysis (default: {SERVER_DEFAULT_REFRESH_MINUTES:g})",
    )
    args = parser.parse_args()

    if args.since_date and args.until_date and args.since_date > args.until_date:
        parser.error("--since must be on or before --until")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.refresh_minutes <= 0:
        parser.error("--refresh-minutes must be greater than 0")

    inputs = {
        source: path.expanduser().resolve()
        for source, path in inputs_from_args(args).items()
    }

    print("Analysing...", flush=True)

    # Importing the reporting stack loads pandas, NumPy, and Matplotlib. Keep
    # that work after argument parsing so --help is instant and users see
    # progress before the heavier modules load.
    matplotlib_cache_dir = Path.home() / ".agentrecap" / "cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    if not any(matplotlib_cache_dir.glob("fontlist*.json")):
        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            for bundled_cache in (Path(bundle_dir) / "agentrecap").glob("fontlist*.json"):
                shutil.copy2(bundled_cache, matplotlib_cache_dir / bundled_cache.name)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
    from .report import build_report, run_pipeline

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

    def build(build_id: int) -> Path:
        """Regenerate the whole report directory. Reused for every server rerun."""
        run_pipeline(inputs, output_dir, start_time=start_time, end_time=end_time)
        return build_report(
            output_dir,
            args.title,
            server=(
                {"build": build_id, "refresh_label": refresh_label(args.refresh_minutes)}
                if args.server
                else None
            ),
        )

    try:
        index_path = build(1)
    except ValueError as error:
        parser.error(str(error))

    print(f'Generated report at "{index_path}"')

    if args.server:
        serve(
            build,
            output_dir,
            port=args.port,
            minutes=args.refresh_minutes,
            open_browser=args.open,
            build_id=1,
        )
        return

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
