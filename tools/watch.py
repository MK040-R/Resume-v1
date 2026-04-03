#!/usr/bin/env python3
"""
watch.py — Watch raw/ for new files and auto-trigger compile

Usage:
    python tools/watch.py              # watch in foreground
    python tools/watch.py --daemon     # run as background daemon
    python tools/watch.py --stop       # stop daemon (if running)
    python tools/watch.py --status     # check if daemon is running
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Timer

sys.path.insert(0, str(Path(__file__).parent))
from utils import REPO_ROOT, SUPPORTED_EXTENSIONS, console, load_config, resolve_paths

PID_FILE = REPO_ROOT / ".watch.pid"
WATCH_LOG = REPO_ROOT / "watch.log"


# ── Logging ───────────────────────────────────────────────────────────────────

def log(message: str) -> None:
    """Write a timestamped message to watch.log and print to console."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{now}] {message}"
    with WATCH_LOG.open("a") as f:
        f.write(line + "\n")
    console.print(f"[dim]{line}[/dim]")


# ── Compile trigger ───────────────────────────────────────────────────────────

class DebouncedCompiler:
    """
    Debounces file system events and triggers compile.py after a quiet period.
    """

    def __init__(self, debounce_seconds: float) -> None:
        self.debounce_seconds = debounce_seconds
        self._timer: Timer | None = None
        self._pending_files: set[str] = set()

    def schedule(self, filepath: str) -> None:
        """Schedule a compile run, debouncing rapid file additions."""
        self._pending_files.add(filepath)
        if self._timer is not None:
            self._timer.cancel()
        self._timer = Timer(self.debounce_seconds, self._run_compile)
        self._timer.daemon = True
        self._timer.start()

    def _run_compile(self) -> None:
        files = list(self._pending_files)
        self._pending_files.clear()
        self._timer = None

        log(f"New/modified file(s) detected: {', '.join(Path(f).name for f in files)}")
        log("Triggering incremental compile...")

        compile_script = REPO_ROOT / "tools" / "compile.py"
        result = subprocess.run(
            [sys.executable, str(compile_script)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            log("Compile completed successfully")
        else:
            log(f"Compile failed (exit code {result.returncode})")
            if result.stderr:
                log(f"Stderr: {result.stderr[:500]}")


# ── File system event handler ─────────────────────────────────────────────────

def make_event_handler(raw_dir: Path, compiler: DebouncedCompiler):
    """Create a watchdog event handler for the raw/ directory."""
    try:
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        raise ImportError("watchdog not installed. Run: pip install watchdog")

    class RawDirHandler(FileSystemEventHandler):
        def _should_handle(self, path: str) -> bool:
            p = Path(path)
            return (
                p.parent.resolve() == raw_dir.resolve()
                and not p.name.startswith(".")
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )

        def on_created(self, event):
            if not event.is_directory and self._should_handle(event.src_path):
                log(f"New file detected: {Path(event.src_path).name}")
                compiler.schedule(event.src_path)

        def on_modified(self, event):
            if not event.is_directory and self._should_handle(event.src_path):
                log(f"File modified: {Path(event.src_path).name}")
                compiler.schedule(event.src_path)

        def on_moved(self, event):
            if not event.is_directory and self._should_handle(event.dest_path):
                log(f"File moved in: {Path(event.dest_path).name}")
                compiler.schedule(event.dest_path)

    return RawDirHandler()


# ── Watcher main loop ─────────────────────────────────────────────────────────

def run_watcher(raw_dir: Path, debounce_seconds: float) -> None:
    """Start the file watcher in the current process."""
    try:
        from watchdog.observers import Observer
    except ImportError:
        raise ImportError("watchdog not installed. Run: pip install watchdog")

    compiler = DebouncedCompiler(debounce_seconds)
    handler = make_event_handler(raw_dir, compiler)

    observer = Observer()
    observer.schedule(handler, str(raw_dir), recursive=False)
    observer.start()

    log(f"Watching raw/ for changes (debounce: {debounce_seconds}s) ...")
    console.print(f"[green]Watcher started.[/green] Monitoring: [cyan]{raw_dir}[/cyan]")
    console.print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if not observer.is_alive():
                log("Observer thread died unexpectedly. Restarting...")
                observer.stop()
                observer = Observer()
                observer.schedule(handler, str(raw_dir), recursive=False)
                observer.start()
    except KeyboardInterrupt:
        log("Watcher stopped by user (Ctrl+C)")
        console.print("\n[yellow]Watcher stopped.[/yellow]")
    finally:
        observer.stop()
        observer.join()


# ── Daemon support ────────────────────────────────────────────────────────────

def start_daemon(raw_dir: Path, debounce_seconds: float) -> None:
    """Fork and run the watcher as a background daemon."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            console.print(f"[yellow]Watcher daemon already running (PID {pid}).[/yellow]")
            console.print("Use --stop to stop it first, or --status to check.")
            return
        except (ProcessLookupError, ValueError):
            PID_FILE.unlink(missing_ok=True)

    # Double-fork to daemonize
    pid = os.fork()
    if pid > 0:
        # Parent: wait briefly then report
        time.sleep(0.5)
        if PID_FILE.exists():
            daemon_pid = int(PID_FILE.read_text().strip())
            console.print(f"[green]Watcher daemon started (PID {daemon_pid}).[/green]")
            console.print(f"Log: [cyan]{WATCH_LOG}[/cyan]")
        return

    # First child: detach from terminal
    os.setsid()
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    # Grandchild: actual daemon process
    # Redirect stdio
    with open(os.devnull, "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    with open(str(WATCH_LOG), "a") as logfile:
        os.dup2(logfile.fileno(), sys.stdout.fileno())
        os.dup2(logfile.fileno(), sys.stderr.fileno())

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    def cleanup(signum, frame):
        log("Daemon received shutdown signal")
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        run_watcher(raw_dir, debounce_seconds)
    finally:
        PID_FILE.unlink(missing_ok=True)

    os._exit(0)


def stop_daemon() -> None:
    """Stop the running watcher daemon."""
    if not PID_FILE.exists():
        console.print("[yellow]No watcher daemon is running (no PID file found).[/yellow]")
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]Watcher daemon (PID {pid}) stopped.[/green]")
    except (ProcessLookupError, ValueError):
        console.print("[yellow]Daemon process not found. Cleaning up PID file.[/yellow]")
        PID_FILE.unlink(missing_ok=True)
    except PermissionError:
        console.print(f"[red]Permission denied when stopping PID. Try: kill {PID_FILE.read_text().strip()}[/red]")


def status_daemon() -> None:
    """Check whether the watcher daemon is running."""
    if not PID_FILE.exists():
        console.print("[yellow]Watcher daemon is NOT running.[/yellow]")
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        console.print(f"[green]Watcher daemon is running (PID {pid}).[/green]")
        console.print(f"Log: [cyan]{WATCH_LOG}[/cyan]")
    except (ProcessLookupError, ValueError):
        console.print("[yellow]Stale PID file found. Daemon is NOT running.[/yellow]")
        PID_FILE.unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Watch raw/ and auto-compile on changes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--daemon", action="store_true", help="Run as background daemon")
    group.add_argument("--stop", action="store_true", help="Stop the running daemon")
    group.add_argument("--status", action="store_true", help="Check daemon status")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
        return
    if args.status:
        status_daemon()
        return

    config = load_config()
    paths = resolve_paths(config)
    raw_dir = paths["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    debounce = config["watch"]["debounce_seconds"]

    console.rule("[bold blue]Knowledge Base Watcher[/bold blue]")

    if args.daemon:
        if sys.platform == "win32":
            console.print("[red]Daemon mode not supported on Windows. Run without --daemon.[/red]")
            sys.exit(1)
        start_daemon(raw_dir, debounce)
    else:
        run_watcher(raw_dir, debounce)


if __name__ == "__main__":
    main()
