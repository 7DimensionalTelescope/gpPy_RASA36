#!/usr/bin/env python3
"""
rasa36_reprocess.py — Batch reprocessing for RASA36.

1. Extracts any tar in /data6/obsdata/RASA36/temp_tar/ that has no extracted
   folder yet (or an empty one) into /data6/obsdata/RASA36/image/.
2. Runs IMSNG_Routine_RASA36.py sequentially (one at a time, oldest first) on
   the selected folders under /data6/obsdata/RASA36/image/.

After each completed run (success or failure) the folder is appended as
"<folder>.tar" to log/rasa36_watcher_processed.txt so rasa36_watcher.py never
re-runs the same tar. Runs interrupted by a stop signal stay unmarked.

Usage (rasa env; long batches should run under nohup/tmux):
    conda run -n rasa python rasa36_reprocess.py --dry-run
    nohup conda run -n rasa python rasa36_reprocess.py > log/reprocess_$(date +%%Y%%m%%dT%%H%%M).out 2>&1 &

Stopping:
    kill -TERM <pid>   # graceful: finish the current folder, then stop
    Ctrl+C / 2nd TERM  # abort the in-flight pipeline run too (left unmarked)

Safety on this shared server: strictly sequential (never more than one
pipeline child, --ncores default 4), refuses to start while rasa36_watcher.py
is running (same-folder collision risk; override with --allow-watcher), and a
pidfile prevents two reprocess instances.
"""

import argparse
import fnmatch
import logging
import os
import signal
import subprocess
import sys
import tarfile
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and constants (shared with rasa36_watcher.py)
# ---------------------------------------------------------------------------
PIPELINE_DIR    = Path(__file__).parent.resolve()
TAR_DIR         = Path("/data6/obsdata/RASA36/temp_tar")
EXTRACT_DIR     = Path("/data6/obsdata/RASA36")
IMAGE_DIR       = EXTRACT_DIR / "image"
PIPELINE_SCRIPT = PIPELINE_DIR / "IMSNG_Routine_RASA36.py"
KEYS_FILE       = PIPELINE_DIR / "keys.dat"
STATE_FILE      = PIPELINE_DIR / "log" / "rasa36_watcher_processed.txt"
PID_FILE        = PIPELINE_DIR / "log" / "rasa36_reprocess.pid"
LOG_FILE        = PIPELINE_DIR / "log" / "rasa36_reprocess.log"

SLACK_CHANNEL   = "#pipeline--imsng"
DEFAULT_NCORES  = 4
STALL_ALERT_SEC = 3600   # Slack alert when the run's log has been quiet this long (alert-only)
STALL_POLL_SEC  = 60
FITS_SUFFIXES   = (".fits", ".fit", ".fts")

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
    logger = logging.getLogger("rasa36_reprocess")
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
# Slack notifications (same channel/keys as the watcher)
# ---------------------------------------------------------------------------
_slack_token: str | None = None
_slack_enabled = True


def notify(text: str, thread_ts: str | None = None) -> str | None:
    """Send a Slack message. Returns the message ts (for threading); None on failure."""
    global _slack_token
    if not _slack_enabled:
        return None
    if _slack_token is None:
        try:
            from astropy.io import ascii  # type: ignore
            keytbl = ascii.read(str(KEYS_FILE))
            _slack_token = keytbl["key"][keytbl["name"] == "slack"].item()
        except Exception as exc:
            log.warning(f"Could not load Slack token from {KEYS_FILE}: {exc}")
            return None
    try:
        from util import tool  # type: ignore
        return tool.slack_bot(_slack_token, SLACK_CHANNEL, text, thread_ts=thread_ts)
    except Exception as exc:
        log.warning(f"Slack notification failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Signal handling — TERM/first Ctrl+C: finish current folder then stop;
# a second signal kills the in-flight pipeline child.
# ---------------------------------------------------------------------------
_stop_requested = False
_current_proc: subprocess.Popen | None = None


def _handle_signal(signum, _frame) -> None:
    global _stop_requested
    if _stop_requested:
        if _current_proc is not None and _current_proc.poll() is None:
            log.warning(f"Second signal — terminating pipeline child PID={_current_proc.pid}")
            _current_proc.terminate()
        return
    _stop_requested = True
    log.info(
        f"Received signal {signum} — will stop after the current folder. "
        "(Note: Ctrl+C in a terminal also SIGINTs the pipeline child directly.)"
    )


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_fits(folder: Path) -> int:
    try:
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in FITS_SUFFIXES)
    except OSError:
        return 0


def _load_processed() -> set[str]:
    if STATE_FILE.exists():
        return set(STATE_FILE.read_text().splitlines())
    return set()


def _mark_processed(folder_name: str) -> None:
    """Record completion under the watcher's key convention (<folder>.tar)."""
    entry = folder_name + ".tar"
    if entry in _load_processed():
        return
    with STATE_FILE.open("a") as fh:
        fh.write(entry + "\n")


def _watcher_running() -> bool:
    out = subprocess.run(
        ["pgrep", "-f", "rasa36_watcher.py"], capture_output=True, text=True
    )
    return bool(out.stdout.strip())


def _same_folder_run_active(folder: Path) -> bool:
    """True if a pipeline is already running on this input folder (forbidden overlap)."""
    out = subprocess.run(
        ["pgrep", "-f", f"IMSNG_Routine_RASA36.py {folder}"],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def _acquire_pidfile() -> bool:
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)   # raises if the process is gone
            log.error(f"Another rasa36_reprocess.py is running (PID {old_pid}) — aborting.")
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            log.warning(f"Stale pidfile {PID_FILE} — replacing.")
    PID_FILE.write_text(str(os.getpid()))
    return True


# ---------------------------------------------------------------------------
# Step 1 — extract not-yet-extracted tars
# ---------------------------------------------------------------------------
def _tars_needing_extraction() -> list[Path]:
    pending = []
    for tar_path in sorted(TAR_DIR.glob("*.tar"), key=lambda p: p.name):
        target = IMAGE_DIR / tar_path.stem
        if not target.is_dir() or _count_fits(target) == 0:
            pending.append(tar_path)
    return pending


def _extract_tar(tar_path: Path, thread_ts: str | None = None) -> Path | None:
    """Extract one tar into EXTRACT_DIR (same layout logic as the watcher).
    Returns the extracted data directory, or None on failure."""
    log.info(f"Extracting {tar_path.name} → {EXTRACT_DIR}")
    notify(f":package: Extracting `{tar_path.name}` → `{EXTRACT_DIR}` …", thread_ts=thread_ts)
    try:
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
        extracted = EXTRACT_DIR / data_relpath
        n_fits = _count_fits(extracted)
        log.info(f"Extraction complete: {extracted} ({n_fits} FITS files)")
        notify(
            f":white_check_mark: Extracted `{tar_path.name}` → `{extracted}` ({n_fits} FITS files)",
            thread_ts=thread_ts,
        )
        return extracted
    except Exception as exc:
        log.error(f"Extraction failed for {tar_path.name}: {exc}")
        notify(f":x: *Extraction failed* — `{tar_path.name}`\n{exc}", thread_ts=thread_ts)
        return None


# ---------------------------------------------------------------------------
# Step 2 — sequential pipeline runs
# ---------------------------------------------------------------------------
def _newest_log_mtime(input_folder: str) -> float | None:
    logdir = PIPELINE_DIR / "log" / "RASA36" / input_folder
    try:
        logs = list(logdir.glob("*.log"))
        return max(p.stat().st_mtime for p in logs) if logs else None
    except OSError:
        return None


def _wait_with_stall_alert(proc, input_folder: str, thread_ts: str | None) -> None:
    """proc.wait() plus alert-only stall detection (same logic as the watcher)."""
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


def _run_pipeline(folder: Path, ncores: int, thread_ts: str | None) -> int:
    """Run the pipeline on one folder. Returns the child's return code."""
    global _current_proc
    cmd = [sys.executable, str(PIPELINE_SCRIPT), str(folder), str(ncores)]
    log.info(f"Starting pipeline: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(PIPELINE_DIR))
    _current_proc = proc
    log.info(f"Pipeline running (PID={proc.pid})")
    try:
        _wait_with_stall_alert(proc, folder.name, thread_ts)
    finally:
        _current_proc = None
    return proc.returncode


def _select_folders(args, processed: set[str]) -> list[Path]:
    folders = sorted(
        (p for p in IMAGE_DIR.iterdir() if p.is_dir()), key=lambda p: p.name
    )
    if args.only:
        folders = [f for f in folders
                   if any(fnmatch.fnmatch(f.name, pat) for pat in args.only)]
    if args.exclude:
        folders = [f for f in folders
                   if not any(fnmatch.fnmatch(f.name, pat) for pat in args.exclude)]
    if args.start_from:
        folders = [f for f in folders if f.name >= args.start_from]
    if args.skip_processed:
        folders = [f for f in folders if f.name + ".tar" not in processed]
    return folders


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract pending RASA36 tars and reprocess image folders sequentially."
    )
    parser.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                        help=f"CPU cores per pipeline run (default: {DEFAULT_NCORES}; shared server — do not raise casually)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the extraction + run plan and exit (no Slack, nothing runs)")
    parser.add_argument("--only", nargs="+", metavar="GLOB",
                        help="process only folders matching these glob pattern(s), e.g. '2026-06-*'")
    parser.add_argument("--exclude", nargs="+", metavar="GLOB",
                        help="skip folders matching these glob pattern(s)")
    parser.add_argument("--start-from", metavar="NAME",
                        help="skip folders sorting before NAME (resume an interrupted batch)")
    parser.add_argument("--skip-processed", action="store_true",
                        help="skip folders already listed in rasa36_watcher_processed.txt")
    parser.add_argument("--extract-only", action="store_true",
                        help="only extract pending tars, run no pipelines")
    parser.add_argument("--no-extract", action="store_true",
                        help="skip the tar-extraction step")
    parser.add_argument("--no-slack", action="store_true",
                        help="disable Slack notifications")
    parser.add_argument("--allow-watcher", action="store_true",
                        help="proceed even if rasa36_watcher.py is running (risk: same-folder overlap)")
    args = parser.parse_args()

    global _slack_enabled
    _slack_enabled = not (args.no_slack or args.dry_run)

    processed = _load_processed()
    pending_tars = [] if args.no_extract else _tars_needing_extraction()
    folders = [] if args.extract_only else _select_folders(args, processed)

    # ---- dry run: report the plan and exit -------------------------------
    if args.dry_run:
        print(f"Would extract {len(pending_tars)} tar(s):")
        for t in pending_tars:
            print(f"  {t.name}")
        print(f"Would run the pipeline on {len(folders)} folder(s) (ncores={args.ncores}):")
        for f in folders:
            mark = " [in processed.txt]" if f.name + ".tar" in processed else ""
            print(f"  {f.name}  ({_count_fits(f)} FITS){mark}")
        return 0

    # ---- catch a mistyped --only/--start-from before anything (incl. Slack) fires --------
    if not args.extract_only and not folders and (args.only or args.start_from):
        log.error(
            "No folders matched --only/--start-from — check for a typo "
            f"(--only={args.only!r}, --start-from={args.start_from!r}). Nothing to do; not notifying Slack."
        )
        return 1

    # ---- safety gates -----------------------------------------------------
    if not _acquire_pidfile():
        return 1
    try:
        if _watcher_running() and not args.allow_watcher:
            log.error(
                "rasa36_watcher.py is running — stop it first, or pass --allow-watcher "
                "(same-folder overlap wipes the run's factory dir)."
            )
            return 1

        # Memory watchdog intentionally NOT spawned here: reprocess runs strictly
        # sequentially (one folder / 4 cores), so its memory footprint is far below
        # the watcher's concurrent lanes. Start memory_watchdog.py separately if you
        # want monitoring during a sweep.
        start_time = time.strftime("%Y-%m-%d %H:%M:%S (%Z)")
        thread_ts = notify("\n".join([
            ":recycle: *RASA36 reprocess started*",
            f"Extract: {len(pending_tars)} tar(s) | Run: {len(folders)} folder(s) | "
            f"Cores: {args.ncores} | Start: {start_time}",
        ]))

        # ---- step 1: extraction ------------------------------------------
        for tar_path in pending_tars:
            if _stop_requested:
                break
            _extract_tar(tar_path, thread_ts=thread_ts)
        # Folders that just appeared should be picked up in this same batch
        if pending_tars and not args.extract_only:
            folders = _select_folders(args, _load_processed())

        # ---- step 2: sequential pipeline runs -----------------------------
        ok, failed, skipped = [], [], []
        for i, folder in enumerate(folders, 1):
            if _stop_requested:
                log.info(f"Stop requested — {len(folders) - i + 1} folder(s) left unprocessed.")
                skipped.extend(f.name for f in folders[i - 1:])
                break
            if _count_fits(folder) == 0:
                log.warning(f"[{i}/{len(folders)}] {folder.name}: no FITS files — skipping.")
                skipped.append(folder.name)
                continue
            if _same_folder_run_active(folder):
                log.error(f"[{i}/{len(folders)}] {folder.name}: a pipeline is already "
                          "running on this folder — skipping (left unmarked).")
                notify(f":no_entry: `{folder.name}` skipped — another run on it is active.",
                       thread_ts=thread_ts)
                skipped.append(folder.name)
                continue

            log.info(f"[{i}/{len(folders)}] Reprocessing {folder.name}")
            notify(f":hammer_and_wrench: [{i}/{len(folders)}] Pipeline starting — `{folder.name}`",
                   thread_ts=thread_ts)
            rc = _run_pipeline(folder, args.ncores, thread_ts)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S (%Z)")

            if rc == 0:
                ok.append(folder.name)
                log.info(f"[{i}/{len(folders)}] {folder.name}: pipeline OK")
                notify(f":white_check_mark: [{i}/{len(folders)}] `{folder.name}` done ({end_time})",
                       thread_ts=thread_ts)
                _mark_processed(folder.name)
            elif _stop_requested:
                # child was interrupted, not a real failure — leave unmarked
                log.warning(f"[{i}/{len(folders)}] {folder.name}: interrupted (rc={rc}), left unmarked.")
                skipped.append(folder.name)
                notify(f":octagonal_sign: [{i}/{len(folders)}] `{folder.name}` interrupted (rc={rc})",
                       thread_ts=thread_ts)
            else:
                failed.append(folder.name)
                log.error(f"[{i}/{len(folders)}] {folder.name}: pipeline FAILED (rc={rc})")
                notify(f":x: [{i}/{len(folders)}] `{folder.name}` FAILED (rc={rc}) ({end_time})",
                       thread_ts=thread_ts)
                _mark_processed(folder.name)   # watcher semantics: completed = won't auto-rerun

        # ---- summary -------------------------------------------------------
        lines = [
            ":checkered_flag: *RASA36 reprocess finished*" if not _stop_requested
            else ":octagonal_sign: *RASA36 reprocess stopped early*",
            f"OK: {len(ok)} | Failed: {len(failed)} | Skipped: {len(skipped)}",
        ]
        if failed:
            lines.append("Failed: " + ", ".join(f"`{n}`" for n in failed))
            lines.append("Re-run failures with: `rasa36_reprocess.py --only " + " ".join(failed) + "`")
        summary = "\n".join(lines)
        log.info(summary.replace("*", "").replace("`", ""))
        notify(summary, thread_ts=thread_ts)
        notify(lines[0] + f" — OK {len(ok)}, failed {len(failed)}, skipped {len(skipped)}")

        return 0 if not failed else 2
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
