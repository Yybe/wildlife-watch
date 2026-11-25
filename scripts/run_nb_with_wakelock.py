"""Run a command while holding a wake lock, so idle sleep cannot kill a multi-hour training job.

Why this exists: on 2026-10-24 a 30-epoch run died at epoch 11 when the laptop power-cycled at 11:04.
Nothing in the code failed - the process tree simply stopped existing. A wake lock does not prevent a
deliberate shutdown, but it does prevent the timeout/lid-close case from silently destroying hours of
work, and it records the child's exit code so a death is never mistaken for a finish.

Run:  .venv/Scripts/python.exe DL_PBL/scripts/run_nb_with_wakelock.py <cwd> <logfile> <cmd...>
"""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def hold(on: bool) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main() -> int:
    cwd, log, cmd = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} launch: {' '.join(cmd)}\n")
        fh.write(f"{time.strftime('%H:%M:%S')} wake lock held (prevents idle sleep only)\n")
        hold(True)
        try:
            proc = subprocess.run(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=os.environ)
        finally:
            hold(False)
        secs = round(time.time() - t0, 1)
        fh.write(f"{time.strftime('%H:%M:%S')} exit={proc.returncode} after {secs}s\n")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
