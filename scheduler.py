"""
scheduler.py — runs calculator.py every weekday morning at a chosen time.
Uses Python's built-in `schedule` library (pip install schedule).

Usage:
    python scheduler.py            # defaults to 07:00
    python scheduler.py --time 06:30
"""

import schedule
import time
import subprocess
import sys
import os
import argparse
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CALC_SCRIPT = os.path.join(SCRIPT_DIR, "calculator.py")
LOG_FILE    = os.path.join(SCRIPT_DIR, "scheduler.log")


def log(msg):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_calculation():
    today = datetime.today().weekday()   # 0=Mon … 4=Fri
    if today >= 5:
        log("Weekend — skipping.")
        return

    log("Starting OE calculation run…")
    try:
        result = subprocess.run(
            [sys.executable, CALC_SCRIPT],
            capture_output=True, text=True, timeout=1800  # 30 min max
        )
        if result.returncode == 0:
            log("Calculation completed successfully.")
        else:
            log(f"ERROR: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        log("TIMEOUT: calculation exceeded 30 minutes.")
    except Exception as e:
        log(f"EXCEPTION: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="07:00",
                        help="Daily run time in HH:MM 24hr format (default: 07:00)")
    args = parser.parse_args()

    run_time = args.time
    log(f"Scheduler started. Will run every weekday at {run_time}.")
    log(f"Calculator: {CALC_SCRIPT}")
    log(f"Log file:   {LOG_FILE}")

    schedule.every().day.at(run_time).do(run_calculation)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
