#!/usr/bin/env python3
"""
rasa36_watcher.py — Watches /data6/obsdata/RASA36/temp_tar/ for new tar files,
extracts them to /data6/obsdata/RASA36/, and runs IMSNG_Routine_RASA36.py on
each extracted directory.

Uses inotify (watchdog >= 2.1) for instant detection:
  Primary  — on_closed on .tar  (IN_CLOSE_WRITE: fires when hpnscp closes the FD)
  Fallback — on_created on .done (SSH sentinel written after successful transfer)
A deduplication set ensures the pipeline runs exactly once per tar regardless of
which event fires first.  Requires: pip install "watchdog>=2.1.0"

Slack threading: each tar file gets its own Slack thread. The initial detection
message is the parent; extraction, pipeline start/end messages are replies.

Two pipeline lanes: tars whose stem ends in `_ToO` go to a dedicated "too" lane
that runs in parallel with the regular lane, so a ToO never waits behind a
nightly reduction. Each lane is a single worker thread (jobs within a lane are
still serialized). Safe because each run works in its own factory subdir
(factory/RASA36/<input_folder>/) and master-frame writes are atomic.
"""

import argparse
import logging
import os
import queue
import signal
import subprocess
import sys
import tarfile
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from watchdog.observers.inotify import InotifyObserver
from watchdog.events import FileSystemEventHandler

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PIPELINE_DIR    = Path(__file__).parent.resolve()
WATCH_DIR       = Path("/data6/obsdata/RASA36/temp_tar")
EXTRACT_DIR     = Path("/data6/obsdata/RASA36")
PIPELINE_SCRIPT = PIPELINE_DIR / "IMSNG_Routine_RASA36.py"
WATCHDOG_SCRIPT = PIPELINE_DIR / "memory_watchdog.py"
KEYS_FILE       = PIPELINE_DIR / "keys.dat"
STATE_FILE      = PIPELINE_DIR / "log" / "rasa36_watcher_processed.txt"
LOG_FILE        = PIPELINE_DIR / "log" / "rasa36_watcher.log"

SLACK_CHANNEL  = "#pipeline--imsng"
MAX_TAR_FILES: int | None = None   # delete oldest tars when count exceeds this; None = keep all
DEFAULT_NCORES = 4
STALL_ALERT_SEC = 3600   # Slack alert when the run's log has been quiet this long (alert-only)
STALL_POLL_SEC  = 60

sys.path.insert(0, str(PIPELINE_DIR))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("rasa36_watcher")
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fh)
    return logger


log = _setup_logging()


# ---------------------------------------------------------------------------
# Slack notifications
# ---------------------------------------------------------------------------
_slack_token: str | None = None


def _load_slack_token() -> str | None:
    try:
        from astropy.io import ascii  # type: ignore
        keytbl = ascii.read(str(KEYS_FILE))
        return keytbl["key"][keytbl["name"] == "slack"].item()
    except Exception as exc:
        log.warning(f"Could not load Slack token from {KEYS_FILE}: {exc}")
        return None


def notify(text: str, thread_ts: str | None = None) -> str | None:
    """Send a Slack message. Returns the message ts (for threading); None on failure."""
    global _slack_token
    if _slack_token is None:
        _slack_token = _load_slack_token()
    if _slack_token is None:
        log.debug("Slack token unavailable — skipping notification")
        return None
    try:
        from util import tool  # type: ignore
        return tool.slack_bot(_slack_token, SLACK_CHANNEL, text, thread_ts=thread_ts)
    except Exception as exc:
        log.warning(f"Slack notification failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Processed-file state
# ---------------------------------------------------------------------------
def _load_processed() -> set[str]:
    if STATE_FILE.exists():
        return set(STATE_FILE.read_text().splitlines())
    return set()


def _mark_processed(tar_name: str, processed: set[str]) -> None:
    processed.add(tar_name)
    with STATE_FILE.open("a") as fh:
        fh.write(tar_name + "\n")


# ---------------------------------------------------------------------------
# Tar file cleanup — keep at most MAX_TAR_FILES
# ---------------------------------------------------------------------------
def _cleanup_old_tars(thread_ts: str | None = None) -> None:
    if MAX_TAR_FILES is None:
        return
    tars = sorted(WATCH_DIR.glob("*.tar"), key=lambda p: p.stat().st_mtime)
    for tar in tars[: max(0, len(tars) - MAX_TAR_FILES)]:
        log.info(f"Deleting old tar (limit={MAX_TAR_FILES}): {tar.name}")
        notify(
            f":wastebasket: Deleted old tar to stay within limit ({MAX_TAR_FILES}): `{tar.name}`",
            thread_ts=thread_ts,
        )
        tar.unlink()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_running = True

# In-flight pipeline subprocesses, so a second Ctrl+C can force-kill them
# directly instead of waiting on thread joins that may never return (e.g. a
# wedged worker, or a subprocess not in this terminal's process group).
_active_procs: set[subprocess.Popen] = set()
_active_procs_lock = threading.Lock()
_watchdog_proc: subprocess.Popen | None = None


def _handle_signal(signum, _frame) -> None:
    global _running
    if _running:
        # First signal: flip the flag *before* the (blocking, network) notify()
        # call below — otherwise a second Ctrl+C arriving while notify() is
        # still in flight would see _running still True and re-enter this
        # branch, sending a duplicate "stopped" message to Slack.
        _running = False
        log.info(
            f"Received signal {signum} — shutting down. Queued pipelines will not be "
            "started (they stay unmarked and re-queue at the next watcher start). "
            "The in-flight run gets the same signal; press Ctrl+C again to force-kill immediately."
        )
        notify(
            ":octagonal_sign: *RASA36 Watcher stopping* — finishing the in-flight "
            "pipeline run, no new runs will start. Press Ctrl+C again to force-kill now."
        )
        return

    # Second signal while already shutting down: don't wait on the in-flight
    # run any longer — kill it directly and exit immediately. Covers a wedged
    # worker (see toYP_pipeline/CLAUDE.md known failure signatures) and the
    # case where the pipeline subprocess never saw the first signal at all
    # (e.g. watcher launched under nohup/tmux/systemd, different process group).
    log.warning(f"Received signal {signum} again — force-killing in-flight run(s) and exiting now.")
    notify(":octagonal_sign: *RASA36 Watcher force-stopped* — killed in-flight pipeline run(s).")
    with _active_procs_lock:
        procs = list(_active_procs)
    for proc in procs:
        try:
            proc.kill()
        except Exception as exc:
            log.warning(f"Could not kill PID {proc.pid}: {exc}")
    if _watchdog_proc is not None and _watchdog_proc.poll() is None:
        try:
            _watchdog_proc.kill()
        except Exception:
            pass
    logging.shutdown()
    os._exit(1)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def _extract_tar(tar_path: Path, thread_ts: str | None = None) -> Path:
    """Extract tar into EXTRACT_DIR. Returns the absolute path of the data directory.

    The observatory packs data as image/<stem>/... so the top-level directory
    inside the tar is not necessarily <stem>. We scan the tar members to find the
    directory whose basename matches the tar stem and return its full path after
    extraction (e.g. /data6/obsdata/RASA36/image/2026-06-10_gain25).
    """
    log.info(f"Extracting {tar_path.name} → {EXTRACT_DIR}")
    notify(
        f":package: Extracting `{tar_path.name}` → `{EXTRACT_DIR}` …",
        thread_ts=thread_ts,
    )

    stem = tar_path.stem
    with tarfile.open(tar_path) as tf:
        data_relpath: str | None = None
        for member in tf.getmembers():
            if member.isdir() and Path(member.name).name == stem:
                data_relpath = member.name.rstrip("/")
                break
        if data_relpath is None:
            raise ValueError(f"No directory named '{stem}' found inside {tar_path.name}")
        tf.extractall(EXTRACT_DIR, filter="data")

    extracted_path = EXTRACT_DIR / data_relpath

    # Count FITS files in the extracted directory for the completion message
    try:
        n_fits = sum(
            1 for p in extracted_path.rglob("*")
            if p.suffix.lower() in (".fits", ".fit", ".fts")
        )
    except Exception:
        n_fits = 0

    log.info(f"Extraction complete: {extracted_path} ({n_fits} FITS files)")
    notify(
        "\n".join([
            f":white_check_mark: *Extraction done* — `{tar_path.name}`",
            f"Path: `{extracted_path}` | FITS files: {n_fits}",
        ]),
        thread_ts=thread_ts,
    )
    return extracted_path


def _extracted_dir_complete(tar_path: Path, extracted_dir: Path) -> bool:
    """True if the bytes already on disk under extracted_dir match the tar's total
    file size — i.e. a prior extraction actually finished rather than being cut
    short (watcher restart mid-unpack, disk full, interrupted transfer). A bare
    ">=1 FITS file" check let a partial extraction look 'done' and skip
    re-extraction forever."""
    try:
        with tarfile.open(tar_path) as tf:
            expected_size = sum(m.size for m in tf.getmembers() if m.isfile())
    except Exception as exc:
        log.warning(f"Could not read {tar_path.name} to verify extraction completeness: {exc}")
        return False
    try:
        actual_size = sum(p.stat().st_size for p in extracted_dir.rglob("*") if p.is_file())
    except OSError:
        return False
    return actual_size >= expected_size


MAX_PIPELINE_ATTEMPTS = 2   # 1 initial run + 1 retry on failure


def _newest_log_mtime(input_folder: str) -> float | None:
    """mtime of the newest pipeline log for this input folder, or None if none yet."""
    logdir = PIPELINE_DIR / "log" / "RASA36" / input_folder
    try:
        logs = list(logdir.glob("*.log"))
        if not logs:
            return None
        return max(p.stat().st_mtime for p in logs)
    except OSError:
        return None


def _wait_with_stall_alert(proc, input_folder: str, thread_ts: str | None) -> None:
    """proc.wait() plus alert-only stall detection.

    A wedged pool worker can freeze the pipeline forever with no error output
    (2026-07-13: the 2026-06-11_gain25 stack stage hung >10 h at 40/41 groups).
    While the pipeline runs, watch its newest log file; if it stops growing for
    STALL_ALERT_SEC, send one Slack alert (re-armed if the log grows again).
    Alert-only — never kills anything.
    """
    t_start = time.time()
    alerted = False
    while proc.poll() is None:
        time.sleep(STALL_POLL_SEC)
        mtime = _newest_log_mtime(input_folder)
        quiet = time.time() - (mtime if mtime is not None else t_start)
        if quiet >= STALL_ALERT_SEC:
            if not alerted:
                alerted = True
                log.warning(
                    f"Pipeline log for {input_folder} quiet for {int(quiet / 60)} min (PID={proc.pid})"
                )
                notify(
                    "\n".join([
                        f":warning: *Pipeline may be stalled* — `{input_folder}`",
                        f"No log output for {int(quiet / 60)} min; PID {proc.pid} is still running.",
                        "Possible wedged worker — see 'Known failure signatures' in toYP_pipeline/CLAUDE.md.",
                    ]),
                    thread_ts=thread_ts,
                )
        else:
            alerted = False
    proc.wait()


def _same_folder_run_active(extracted_path: Path) -> bool:
    """True if some pipeline process is already running on this input folder
    (e.g. an orphan surviving a previous watcher instance). Two runs on the
    same folder wipe each other's factory subdir — forbidden."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"IMSNG_Routine_RASA36.py {extracted_path}"],
            capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def _run_pipeline(
    tar_path: Path,
    extracted_path: Path,
    ncores: int,
    thread_ts: str | None = None,
) -> bool:
    """Run the reduction pipeline synchronously, retrying once on failure. Called from the worker thread.

    Returns True when the run completed its attempts (success or final failure) —
    the caller then marks the tar processed. Returns False when the tar should be
    re-attempted at the next watcher start (same-folder run already active, or
    the watcher was shutting down)."""
    stem = tar_path.stem
    cmd = ["python", str(PIPELINE_SCRIPT), str(extracted_path), str(ncores)]

    if _same_folder_run_active(extracted_path):
        log.error(f"Skipping {stem}: a pipeline is already running on {extracted_path}")
        notify(
            "\n".join([
                f":no_entry: *Pipeline skipped* — `{stem}`",
                f"Another run on `{extracted_path.name}` is already active (orphan from a previous watcher?).",
                "Not marked processed — it will be re-queued at the next watcher restart.",
            ]),
            thread_ts=thread_ts,
        )
        return False

    for attempt in range(1, MAX_PIPELINE_ATTEMPTS + 1):
        if not _running:
            log.info(f"Watcher is shutting down — not starting pipeline for {stem}")
            return False  # leave unmarked so it re-queues at next start
        retry_note = f" (retry {attempt - 1}/{MAX_PIPELINE_ATTEMPTS - 1})" if attempt > 1 else ""
        start_time = time.strftime("%Y-%m-%d %H:%M:%S (%Z)")

        log.info(f"Starting pipeline{retry_note}: {' '.join(cmd)}")
        notify(
            "\n".join([
                f":hammer_and_wrench: *Pipeline starting{retry_note}* — `{extracted_path.name}`",
                f"Start: {start_time} | Cores: {ncores}",
                f"_(pipeline progress will appear in its own Slack thread)_",
            ]),
            thread_ts=thread_ts,
        )

        proc = subprocess.Popen(cmd, cwd=str(PIPELINE_DIR))
        log.info(f"Pipeline running (PID={proc.pid})")
        with _active_procs_lock:
            _active_procs.add(proc)
        try:
            _wait_with_stall_alert(proc, extracted_path.name, thread_ts)
        finally:
            with _active_procs_lock:
                _active_procs.discard(proc)

        rc = proc.returncode
        end_time = time.strftime("%Y-%m-%d %H:%M:%S (%Z)")

        if rc == 0:
            log.info(f"Pipeline finished successfully for {stem}{retry_note}")
            notify(
                "\n".join([
                    f":white_check_mark: *Pipeline completed OK{retry_note}* — `{stem}`",
                    f"End: {end_time}",
                ]),
                thread_ts=thread_ts,
            )
            return True

        log.error(f"Pipeline exited with return code {rc} for {stem}{retry_note}")

        if not _running:
            log.info(f"Watcher is shutting down — skipping retry for {stem}")
            notify(
                "\n".join([
                    f":x: *Pipeline FAILED (watcher shutting down, no retry)* — `{stem}`",
                    f"Return code: {rc} | End: {end_time}",
                ]),
                thread_ts=thread_ts,
            )
            return False   # interrupted by shutdown — leave unmarked so it re-queues

        if attempt < MAX_PIPELINE_ATTEMPTS:
            notify(
                "\n".join([
                    f":warning: *Pipeline failed — retrying* — `{stem}`",
                    f"Return code: {rc} | End: {end_time}",
                ]),
                thread_ts=thread_ts,
            )
        else:
            notify(
                "\n".join([
                    f":x: *Pipeline FAILED after {MAX_PIPELINE_ATTEMPTS} attempts* — `{stem}`",
                    f"Return code: {rc} | End: {end_time}",
                ]),
                thread_ts=thread_ts,
            )
    # All attempts used: mark processed anyway so a permanently failing date
    # does not re-run on every watcher restart (failures are Slack-visible).
    return True


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------
class TarHandler(FileSystemEventHandler):
    """
    Primary  — on_closed on .tar  (IN_CLOSE_WRITE, fires when hpnscp closes the FD)
    Fallback — on_created on .done (SSH sentinel, fires ~1 s later)
    """

    LANES = ("main", "too")   # "too" = fast lane for *_ToO tars, runs in parallel

    def __init__(self, ncores: int, processed: set[str]) -> None:
        self._ncores   = ncores
        self._processed = processed
        self._triggered: set[str] = set()          # within-session deduplication
        self._lock     = threading.Lock()          # guards check-then-add on _triggered
        # Queue items: (tar_path, extracted_path, thread_ts) or None to stop
        self._queues: dict[str, queue.Queue[tuple[Path, Path, str | None] | None]] = {
            lane: queue.Queue() for lane in self.LANES
        }
        self._workers = {
            lane: threading.Thread(
                target=self._pipeline_worker, args=(lane,), daemon=True
            )
            for lane in self.LANES
        }
        for worker in self._workers.values():
            worker.start()

    def stop(self) -> None:
        """Shut down the worker threads. Queued-but-unstarted items are skipped
        (not marked processed) once _running is False, so they re-queue at the
        next watcher start."""
        for q in self._queues.values():
            q.put(None)
        for worker in self._workers.values():
            worker.join()

    def _pipeline_worker(self, lane: str) -> None:
        """One worker per lane — runs that lane's pipelines one at a time."""
        q = self._queues[lane]
        while True:
            item = q.get()
            if item is None:
                break
            tar_path, extracted_path, thread_ts = item
            if not _running:
                log.info(
                    f"Shutdown requested — not starting queued pipeline for {tar_path.name} "
                    "(left unmarked, will re-queue at next watcher start)."
                )
                q.task_done()
                continue
            completed = _run_pipeline(tar_path, extracted_path, self._ncores, thread_ts=thread_ts)
            if completed:
                with self._lock:
                    _mark_processed(tar_path.name, self._processed)
            q.task_done()

    @staticmethod
    def _lane_for(tar_path: Path) -> str:
        return "too" if tar_path.stem.lower().endswith("_too") else "main"

    def on_closed(self, event) -> None:
        if not event.is_directory and event.src_path.endswith(".tar"):
            self._handle(Path(event.src_path))

    def on_created(self, event) -> None:
        if not event.is_directory and event.src_path.endswith(".done"):
            tar_path = Path(event.src_path).with_suffix("")   # strip .done
            if tar_path.exists():
                self._handle(tar_path)
            else:
                log.warning(f"Sentinel arrived but tar not found: {tar_path}")

    def _handle(self, tar_path: Path) -> None:
        with self._lock:
            if tar_path.name in self._triggered or tar_path.name in self._processed:
                return
            self._triggered.add(tar_path.name)

        lane = self._lane_for(tar_path)
        log.info(f"Ready to process: {tar_path.name} (lane={lane})")

        # --- Initial detection message (parent of all follow-up thread replies) ---
        detect_time = time.strftime("%Y-%m-%d %H:%M:%S (%Z)")
        try:
            size_str = f"{tar_path.stat().st_size / 1024**2:.1f} MB"
        except Exception:
            size_str = "N/A"

        lane_note = " :rotating_light: *(ToO — fast lane)*" if lane == "too" else ""
        thread_ts = notify("\n".join([
            f":new: *New tar file detected* — `{tar_path.name}`{lane_note}",
            f"Detected: {detect_time} | Size: {size_str}",
        ]))

        # Cleanup runs after detection so any cleanup messages go into the thread
        _cleanup_old_tars(thread_ts=thread_ts)

        # --- Extraction (skipped when the tar was already unpacked) ---
        existing_dir = EXTRACT_DIR / "image" / tar_path.stem
        n_fits = sum(
            1 for p in existing_dir.rglob("*")
            if p.suffix.lower() in (".fits", ".fit", ".fts")
        ) if existing_dir.is_dir() else 0
        if n_fits > 0 and _extracted_dir_complete(tar_path, existing_dir):
            log.info(f"Already extracted — skipping unpack: {existing_dir} ({n_fits} FITS files)")
            notify(
                f":fast_forward: Already extracted — skipping unpack: `{existing_dir}` ({n_fits} FITS files)",
                thread_ts=thread_ts,
            )
            extracted_path = existing_dir
            self._enqueue(tar_path, extracted_path, lane, thread_ts)
            return
        elif n_fits > 0:
            log.warning(
                f"{existing_dir} has {n_fits} FITS file(s) but total size is smaller "
                "than the tar's contents — extraction looks incomplete, re-extracting."
            )
            notify(
                f":warning: Previous extraction of `{existing_dir}` looks incomplete "
                "(size mismatch vs tar) — re-extracting.",
                thread_ts=thread_ts,
            )

        try:
            extracted_path = _extract_tar(tar_path, thread_ts=thread_ts)
        except Exception as exc:
            log.error(f"Extraction failed for {tar_path.name}: {exc}")
            notify(
                f":x: *Extraction failed* — `{tar_path.name}`\n{exc}",
                thread_ts=thread_ts,
            )
            with self._lock:
                self._triggered.discard(tar_path.name)   # allow retry on next event
            return

        self._enqueue(tar_path, extracted_path, lane, thread_ts)

    def _enqueue(
        self, tar_path: Path, extracted_path: Path, lane: str, thread_ts: str | None
    ) -> None:
        """Remove the tar's sentinel and queue the pipeline run.

        The tar is marked processed only after its pipeline run completes (in
        _pipeline_worker) — marking here at enqueue time meant a watcher restart
        silently dropped every extracted-but-not-yet-reduced tar, because the
        pipeline queue lives only in memory while the startup backlog scan
        trusts the state file.
        """
        done_path = tar_path.with_suffix(".done")
        if done_path.exists():
            done_path.unlink()

        self._queues[lane].put((tar_path, extracted_path, thread_ts))

        # Notify if this job is queued behind others in its lane
        depth = self._queues[lane].qsize()
        if depth > 1:
            log.info(f"Queued for pipeline: {tar_path.name} (lane={lane}, queue depth={depth})")
            notify(
                f":hourglass_flowing_sand: `{tar_path.name}` queued for pipeline ({lane} lane) — position {depth}",
                thread_ts=thread_ts,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(ncores: int) -> None:
    global _watchdog_proc
    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    processed = _load_processed()
    log.info(f"Loaded {len(processed)} previously processed tar(s) from state.")

    handler = TarHandler(ncores, processed)
    observer = InotifyObserver()
    observer.schedule(handler, str(WATCH_DIR), recursive=False)
    observer.start()

    # Memory watchdog (alert-only) — runs as a separate process so it keeps
    # sampling even if the pipeline subprocess hangs or pegs the GIL. The
    # watchdog has its own pidfile lock, so this is a no-op if one is already
    # running (e.g. started manually).
    watchdog_proc = None
    try:
        watchdog_proc = subprocess.Popen(
            ["python", str(WATCHDOG_SCRIPT)], cwd=str(PIPELINE_DIR)
        )
        _watchdog_proc = watchdog_proc
        log.info(f"Memory watchdog launched (PID={watchdog_proc.pid})")
    except Exception as exc:
        log.warning(f"Could not launch memory watchdog: {exc}")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S (%Z)")
    log.info(f"RASA36 watcher started (watch={WATCH_DIR}, ncores={ncores})")
    notify("\n".join([
        f":eyes: *RASA36 Watcher started*",
        f"Watching: `{WATCH_DIR}` | Cores: {ncores} | Start: {start_time}",
    ]))

    # Process any tars that arrived while the watcher was offline
    backlog = sorted(
        [p for p in WATCH_DIR.glob("*.tar") if p.name not in processed],
        key=lambda p: p.stat().st_mtime,
    )
    if backlog:
        log.info(f"Found {len(backlog)} backlog tar(s) at startup.")
        notify(f":inbox_tray: *{len(backlog)} backlog tar(s) found* — processing now.")
        for tar_path in backlog:
            log.info(f"Backlog tar found at startup: {tar_path.name}")
            handler._handle(tar_path)

    try:
        while _running:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
        handler.stop()   # reap the in-flight run (already SIGINT'd); queued runs are skipped
        if watchdog_proc is not None and watchdog_proc.poll() is None:
            watchdog_proc.terminate()
            try:
                watchdog_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watchdog_proc.kill()

    log.info("Watcher exited cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch for RASA36 tar files, extract, and run the reduction pipeline."
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=DEFAULT_NCORES,
        help=f"CPU cores passed to the pipeline (default: {DEFAULT_NCORES})",
    )
    args = parser.parse_args()
    main(args.ncores)
