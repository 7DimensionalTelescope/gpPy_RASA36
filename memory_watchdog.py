#!/usr/bin/env python3
"""
memory_watchdog.py — Polls system memory while the RASA36 reduction pipeline
runs and sends Slack alerts before memory exhaustion makes the server
unresponsive (this happened on 2026-06-29/30: a pipeline run consumed enough
memory that the server became inaccessible until an admin killed it manually).

The real lockup trigger is NOT system-wide RAM: this user's processes run in a
systemd cgroup with MemoryHigh=160 GiB / MemoryMax=180 GiB. Crossing MemoryHigh
makes the kernel throttle and swap *everything this user runs* (pipeline, ssh,
editors) while system RAM still looks free — so this watchdog samples the
cgroup's memory.current against memory.high in addition to /proc/meminfo.

Alert-only: this script never kills or throttles anything. It just gives
early warning so a human can intervene before the server locks up again.

A pidfile-based singleton lock means it's safe to launch this manually and
also have rasa36_watcher.py auto-spawn it on startup — the second instance
just exits.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

PIPELINE_DIR  = Path(__file__).parent.resolve()
KEYS_FILE     = PIPELINE_DIR / "keys.dat"
LOG_FILE      = PIPELINE_DIR / "log" / "memory_watchdog.log"
PID_FILE      = PIPELINE_DIR / "log" / "memory_watchdog.pid"

SLACK_CHANNEL = "#pipeline--imsng"
# The pipeline's memory lives mostly in child processes with their own cmdlines;
# matching only the main script made pipeline_rss report 0.0 GB during real runs.
PROCESS_PATTERNS = (
    "IMSNG_Routine_RASA36.py",
    "solve-field", "astrometry-engine",
    "scamp", "missfits", "SWarp", "swarp",
    "source-extractor", "sex ",
    "hotpants", "wcsremap",
    "gregoryphot", "gregoryfind",
)
CGROUP_DIR = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice")

POLL_SECONDS         = 60
CRIT_AVAIL_PCT       = 5.0     # alert when available memory drops below this % of total
CRIT_SWAP_GB         = 50.0    # alert when swap usage climbs above this (with low RAM)
CRIT_CGROUP_PCT      = 95.0    # alert when user-cgroup usage exceeds this % of memory.high
ALERT_COOLDOWN_SECONDS = 15 * 60   # don't repeat an alert more often than this


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("memory_watchdog")
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fh)
    return logger


log = _setup_logging()


# ---------------------------------------------------------------------------
# Singleton lock — safe to run standalone AND have the watcher auto-spawn it
# ---------------------------------------------------------------------------
def _acquire_singleton_lock() -> bool:
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            os.kill(existing_pid, 0)   # raises if not running
            return False               # another instance is alive
        except (ValueError, ProcessLookupError, PermissionError):
            pass   # stale pidfile — fall through and claim it
    PID_FILE.write_text(str(os.getpid()))
    return True


def _release_singleton_lock() -> None:
    try:
        if PID_FILE.exists() and int(PID_FILE.read_text().strip()) == os.getpid():
            PID_FILE.unlink()
    except Exception:
        pass


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


def notify(text: str) -> None:
    global _slack_token
    if _slack_token is None:
        _slack_token = _load_slack_token()
    if _slack_token is None:
        log.debug("Slack token unavailable — skipping notification")
        return
    try:
        sys.path.insert(0, str(PIPELINE_DIR))
        from util import tool  # type: ignore
        tool.slack_bot(_slack_token, SLACK_CHANNEL, text)
    except Exception as exc:
        log.warning(f"Slack notification failed: {exc}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _read_meminfo() -> tuple[int, int, int]:
    """Returns (total_kb, available_kb, swap_used_kb)."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            info[key] = int(rest.strip().split()[0])
    total_kb = info["MemTotal"]
    avail_kb = info.get("MemAvailable", info["MemFree"])
    swap_used_kb = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    return total_kb, avail_kb, swap_used_kb


def _pipeline_rss_gb() -> tuple[float, int]:
    """Sums RSS across this user's pipeline processes (main script + external children)."""
    import getpass
    me = getpass.getuser()
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,user:32,rss,args"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return 0.0, 0
    total_kb = 0
    n = 0
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4 or parts[1] != me:
            continue
        if not any(pat in parts[3] for pat in PROCESS_PATTERNS):
            continue
        try:
            total_kb += int(parts[2])
            n += 1
        except ValueError:
            continue
    return total_kb / 1024 / 1024, n


def _read_cgroup() -> tuple[float, float] | None:
    """Returns (current_gb, high_gb) for this user's systemd slice, or None.

    memory.high is the threshold where the kernel starts throttling+swapping
    every process this user owns — the practical 'server stops responding'
    limit, even when system-wide RAM is plentiful.
    """
    try:
        current = int((CGROUP_DIR / "memory.current").read_text().strip())
        high_raw = (CGROUP_DIR / "memory.high").read_text().strip()
        if high_raw == "max":
            high_raw = (CGROUP_DIR / "memory.max").read_text().strip()
        if high_raw == "max":
            return None
        return current / 1024**3, int(high_raw) / 1024**3
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_running = True


def _handle_signal(signum, _frame) -> None:
    global _running
    log.info(f"Received signal {signum} — shutting down.")
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    if not _acquire_singleton_lock():
        log.info("Another memory_watchdog instance is already running — exiting.")
        return

    try:
        log.info(f"Memory watchdog started (PID={os.getpid()}, poll={POLL_SECONDS}s).")
        notify(
            ":eyes: *Memory watchdog started* — alert-only, polling every "
            f"{POLL_SECONDS}s. Alerts at <{CRIT_AVAIL_PCT:.0f}% available RAM "
            f"or user-cgroup usage >{CRIT_CGROUP_PCT:.0f}% of its memory.high limit; "
            f"swap alerts only fire when both swap >{CRIT_SWAP_GB:.0f}GB AND available RAM <50%."
        )

        last_alert_level: str | None = None
        last_alert_time = 0.0

        while _running:
            total_kb, avail_kb, swap_used_kb = _read_meminfo()
            avail_pct = 100.0 * avail_kb / total_kb
            avail_gb = avail_kb / 1024 / 1024
            swap_used_gb = swap_used_kb / 1024 / 1024
            pipeline_rss_gb, n_proc = _pipeline_rss_gb()
            cgroup = _read_cgroup()
            cg_pct = 100.0 * cgroup[0] / cgroup[1] if cgroup else None

            # CRITICAL-only (warning tier removed 2026-07-15 — it was noise).
            # Swap uses AND so stale swap (left over from a previous run) does
            # not fire a false alarm when available RAM is actually fine.
            # The user-cgroup check fires independently: it is what actually
            # freezes this user's session, even with system RAM free.
            level = None
            if avail_pct < CRIT_AVAIL_PCT:
                level = "CRITICAL"
            elif cg_pct is not None and cg_pct > CRIT_CGROUP_PCT:
                level = "CRITICAL"
            elif swap_used_gb > CRIT_SWAP_GB and avail_pct < 50.0:
                level = "CRITICAL"

            cg_str = (
                f"cgroup={cgroup[0]:.1f}/{cgroup[1]:.0f}GB ({cg_pct:.0f}%)"
                if cgroup else "cgroup=N/A"
            )
            log.info(
                f"avail={avail_pct:.1f}% ({avail_gb:.1f}GB) | {cg_str} | swap_used={swap_used_gb:.1f}GB | "
                f"pipeline_rss={pipeline_rss_gb:.1f}GB across {n_proc} proc(es)"
                + (f" | LEVEL={level}" if level else "")
            )

            if level:
                now = time.time()
                escalating = last_alert_level is None
                if escalating or (now - last_alert_time > ALERT_COOLDOWN_SECONDS):
                    notify("\n".join([
                        f":rotating_light: *Memory {level}* — available {avail_pct:.1f}% ({avail_gb:.1f} GB), "
                        f"user cgroup {cg_str}, swap used {swap_used_gb:.1f} GB",
                        f"RASA36 pipeline RSS: {pipeline_rss_gb:.1f} GB across {n_proc} process(es)",
                        "_(cgroup near memory.high = kernel will throttle/swap everything this user runs)_",
                        "_(alert-only — nothing was killed; intervene manually if needed)_",
                    ]))
                    last_alert_time = now
                    last_alert_level = level
            elif last_alert_level is not None:
                notify(
                    f":white_check_mark: *Memory back to normal* — "
                    f"available {avail_pct:.1f}%, user cgroup {cg_str}, swap used {swap_used_gb:.1f} GB"
                )
                last_alert_level = None

            for _ in range(POLL_SECONDS):
                if not _running:
                    break
                time.sleep(1)

        log.info("Memory watchdog exited cleanly.")
    finally:
        _release_singleton_lock()


if __name__ == "__main__":
    main()
