"""Serve the report over HTTP and keep rebuilding it in place.

``--server`` keeps agentrecap running instead of exiting after one report. Every
rebuild reuses the same output directory, so the URL stays stable while the
report behind it is refreshed on a timer and from the page's rerun button.

Rebuilds delete and rewrite chart files, so a page request that arrives mid-run
waits for the rebuild to finish rather than serving a half-written directory.
"""

import mimetypes
import socket
import sys
import threading
import traceback
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

# Deliberately outside the usual local-dev range (3000/5000/8000/8080) so the
# server does not collide with whatever else the user is running.
DEFAULT_PORT = 8973
DEFAULT_REFRESH_MINUTES = 30.0
# How long a page request waits on an in-flight rebuild before giving up.
BUILD_WAIT_SECONDS = 300


def refresh_label(minutes: float) -> str:
    """Human-readable rerun interval for the page footer."""
    if minutes >= 60 and minutes % 60 == 0:
        hours = int(minutes // 60)
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{minutes:g} min"


class ReportBuilder:
    """Runs report rebuilds one at a time and tracks the result of the last one.

    ``build`` receives the id of the build it is producing so the rendered page
    can tell the server which report version the browser is currently showing.
    """

    def __init__(self, build: Callable[[int], object], build_id: int = 1) -> None:
        self._build = build
        self._content = threading.Lock()
        self._state = threading.Lock()
        self._running = False
        self._build_id = build_id
        self._generated = _now()
        self._error: str | None = None

    def status(self) -> dict:
        with self._state:
            return {
                "running": self._running,
                "build": self._build_id,
                "generated": self._generated,
                "error": self._error,
            }

    def acquire_content(self, timeout: float) -> bool:
        """Block readers while a rebuild is rewriting the report directory."""
        return self._content.acquire(timeout=timeout)

    def release_content(self) -> None:
        self._content.release()

    def rebuild(self) -> bool:
        """Rebuild the report; returns False when one is already in flight.

        A failed rebuild is recorded and reported in the page, leaving the
        previous report in place: a transient failure should not take down a
        server that has a perfectly good report to serve.
        """
        with self._state:
            if self._running:
                return False
            self._running = True
            build_id = self._build_id + 1

        error: str | None = None
        try:
            with self._content:
                print(f"Rerunning analysis (build {build_id})...", flush=True)
                self._build(build_id)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            traceback.print_exc()
        finally:
            with self._state:
                self._running = False
                self._error = error
                if error is None:
                    self._build_id = build_id
                    self._generated = _now()
                served_build, generated = self._build_id, self._generated

        # The report path never changes, so reruns report when they finished
        # rather than repeating where: silence here leaves anyone watching the
        # terminal unable to tell a slow rebuild from a stuck one.
        if error is None:
            print(f"Report updated {generated} (build {served_build})", flush=True)
        else:
            print(f"Rerun failed, still serving build {served_build}", flush=True)
        return True

    def rebuild_in_background(self) -> bool:
        """Kick off a rebuild without blocking the request that asked for it."""
        with self._state:
            if self._running:
                return False
        threading.Thread(target=self.rebuild, daemon=True).start()
        return True


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def hyperlink(url: str, label: str | None = None) -> str:
    """Make a URL clickable with an OSC 8 escape, or plain when piped to a file.

    Terminals without OSC 8 support ignore the escape and show the label, which
    is the URL itself, so nothing is lost where it is unsupported.
    """
    label = label or url
    if not sys.stdout.isatty():
        return label
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def prompt_to_open(url: str) -> None:
    """Offer to open the report, matching the one-shot run's closing question.

    Skipped when stdin is not a terminal: a backgrounded or scripted server has
    nobody to answer, and it should keep serving rather than sit at a prompt.
    """
    if not sys.stdin.isatty():
        return
    try:
        answer = input("Would you like to open it in the browser? (y/n) ")
    except EOFError:
        print()
        return
    if answer.strip().lower() in {"y", "yes"}:
        webbrowser.open(url)


def create_app(output_dir: Path, builder: ReportBuilder, minutes: float):
    from flask import Flask, Response, abort, jsonify
    from werkzeug.utils import safe_join

    app = Flask(__name__, static_folder=None)

    def serve_file(relative_path: str):
        path = safe_join(str(output_dir), relative_path)
        if path is None:
            abort(404)
        if not builder.acquire_content(BUILD_WAIT_SECONDS):
            return Response(
                "Report rebuild in progress; retry shortly.\n",
                status=503,
                headers={"Retry-After": "5"},
                mimetype="text/plain",
            )
        try:
            file_path = Path(path)
            if not file_path.is_file():
                abort(404)
            # Read the whole file under the lock: rebuilds truncate chart files
            # in place, and a streamed response would outlive the lock.
            payload = file_path.read_bytes()
        finally:
            builder.release_content()
        mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return Response(payload, mimetype=mimetype, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index():
        return serve_file("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({**builder.status(), "refresh_minutes": minutes})

    @app.post("/api/refresh")
    def refresh():
        started = builder.rebuild_in_background()
        return jsonify({**builder.status(), "started": started})

    @app.get("/<path:relative_path>")
    def asset(relative_path: str):
        return serve_file(relative_path)

    return app


def serve(
    build: Callable[[int], object],
    output_dir: Path,
    port: int = DEFAULT_PORT,
    minutes: float = DEFAULT_REFRESH_MINUTES,
    host: str = "127.0.0.1",
    open_browser: bool = False,
    build_id: int = 1,
) -> None:
    """Serve ``output_dir`` until interrupted, rebuilding it every ``minutes``."""
    import logging

    from werkzeug.serving import make_server

    # Per-request logging would bury the rebuild messages that actually matter
    # in a session that stays open for hours.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # Claim the port before the report directory is handed to a server, so a
    # busy port fails with our own message instead of Werkzeug's exit path.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            raise SystemExit(
                f"Could not serve on port {port} ({error}). Use --port to pick another one."
            )

    builder = ReportBuilder(build, build_id=build_id)
    app = create_app(output_dir, builder, minutes)
    server = make_server(host, port, app, threaded=True)

    stop = threading.Event()

    def refresh_loop() -> None:
        while not stop.wait(minutes * 60):
            builder.rebuild()

    threading.Thread(target=refresh_loop, daemon=True).start()
    # Serving runs on its own thread so the prompt below faces a live URL: an
    # answer of "y" opens a page that is ready to load.
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://{host}:{port}/"
    # Flushed because a server that outlives its terminal is often redirected
    # to a log file, where block buffering would hide progress for hours.
    print(f"Serving the report at {hyperlink(url)}", flush=True)
    print(
        f"Rerunning the analysis every {refresh_label(minutes)}. Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        if open_browser:
            webbrowser.open(url)
        else:
            prompt_to_open(url)
        # Ctrl+C reaches this loop; a bare wait can swallow it on some platforms.
        while not stop.wait(1):
            pass
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        server.shutdown()
        print("Stopped serving the report.", flush=True)
