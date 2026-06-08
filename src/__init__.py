"""
Core processing modules for Buddy Core
"""

import os
import sys

# Add project root to sys.path to allow importing from 'shared' and local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sorter import run_sorter
from extractor import run_extractor
from cio import run_cio
from formatter import run_formatter
import subprocess
import time
import argparse
import json
from datetime import datetime
import glob
import pytz
import holidays

from shared.shared_logger import setup_logger

logger = setup_logger(logger_name=__name__)


def is_us_trading_day() -> bool:
    """
    Checks if the current date corresponds to a US trading day.
    US trading days are Monday-Friday excluding NYSE holidays.
    """
    us_tz = pytz.timezone("America/New_York")
    us_now = datetime.now(us_tz).date()

    if us_now.weekday() >= 5:
        return False

    nyse_holidays = holidays.financial_holidays("US", years=us_now.year)
    if us_now in nyse_holidays:
        return False

    return True


def pull_data_from_cloud(report_type: str = "full"):
    """
    Pull sieve's data from oracle cloud storage
    """
    today = datetime.now(pytz.timezone("America/New_York")).strftime("%Y%m%d")

    """
    Change the following variables if you are not using the same environment as mine
    """
    ORACLE_IP_ADDRESS = "159.13.60.28"
    ORACLE_SSH_KEY = "/Users/taehoonkwon/.ssh/oracle-cloud-ssh.key"

    if report_type == "premarket":
        remote_file = f"/home/ubuntu/sieve/data/premarket_news_{today}.json"
        local_dir = "/Users/taehoonkwon/workspaces/buddy/core/data"
        local_file = f"{local_dir}/premarket_news_{today}.json"
    else:
        # Both 'full' and 'incremental' use the daily_news file
        remote_file = f"/home/ubuntu/sieve/data/daily_news_{today}.json"
        local_dir = "/Users/taehoonkwon/workspaces/buddy/core/data"
        local_file = f"{local_dir}/daily_news_{today}.json"

    os.makedirs(local_dir, exist_ok=True)
    scp_command = f"scp -i {ORACLE_SSH_KEY} -o StrictHostKeyChecking=no ubuntu@{ORACLE_IP_ADDRESS}:{remote_file} {local_file}"

    max_retries = 10 if report_type == "full" else 5
    retry_delay = 60  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Pulling data from oracle cloud storage... (Attempt {attempt}/{max_retries})"
            )

            subprocess.run(
                scp_command, shell=True, check=True, capture_output=True, text=True
            )

            if os.path.exists(local_file):
                with open(local_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if report_type == "full":
                    market_map = data.get("market_map")
                    is_empty = False
                    if not market_map:
                        is_empty = True
                    elif isinstance(market_map, dict):
                        indices = market_map.get("Indices")
                        sectors = market_map.get("Sectors")
                        if not indices and not sectors:
                            is_empty = True

                    if is_empty:
                        raise ValueError(
                            f"market_map is empty or missing in {local_file}"
                        )

                logger.info(f"Successfully pulled data: {local_file}")
                return  # Success, exit function
            else:
                raise FileNotFoundError(f"File not found after scp: {local_file}")

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            stderr_msg = ""
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                stderr_msg = f" | Details: {e.stderr.strip()}"

            logger.warning(f"Attempt {attempt} failed: {e}{stderr_msg}")

            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All retries failed.")
                raise


def run_all(report_type: str = "full"):
    """
    Run the entire data processing pipeline sequentially:
    Sorter -> Extractor -> CIO -> Formatter

    After formatter finishes successfully, cleans up all files in the /data directory
    EXCEPT for the final alpha_signal_*.md file.
    """
    # 1. New York Timezone Check
    us_tz = pytz.timezone("America/New_York")
    ny_now = datetime.now(us_tz)
    today_str = datetime.now(us_tz).strftime("%Y%m%d")

    if report_type == "full":
        if ny_now.time() < datetime.strptime("06:00", "%H:%M").time():
            logger.info(
                f"Current NY time ({ny_now.strftime('%H:%M')}) is before 06:00. Skipping execution."
            )
            return
    elif report_type == "premarket":
        if ny_now.time() < datetime.strptime("08:30", "%H:%M").time():
            logger.info(
                f"Current NY time ({ny_now.strftime('%H:%M')}) is before 08:30. Skipping execution."
            )
            return
    elif report_type == "incremental":
        # Incremental runs at 00:00 and 03:00, no time lock strictly needed,
        # but we can let it pass anytime.
        pass

    # 2. Duplicate Execution Prevention (Lock File with Stale/Hung Process Validation)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_file = os.path.join(project_root, "logs", "buddy.lock")

    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as lf:
                lock_pid = int(lf.read().strip())
        except Exception:
            lock_pid = None

        if lock_pid:
            # Check if the process is actually running
            is_running = False
            try:
                os.kill(lock_pid, 0)
                is_running = True
            except OSError:
                is_running = False

            if not is_running:
                logger.info(
                    f"Detected stale lock file (PID {lock_pid} is not running). Cleaning up lock and proceeding."
                )
                try:
                    os.remove(lock_file)
                except Exception:
                    pass
            else:
                # If running, check how long it has been running (stale threshold: 2 hours)
                try:
                    lock_mtime = os.path.getmtime(lock_file)
                    elapsed_hours = (time.time() - lock_mtime) / 3600.0
                except Exception:
                    elapsed_hours = 0.0

                if elapsed_hours >= 2.0:
                    logger.warning(
                        f"Detected hung buddy process (PID {lock_pid}, running for {elapsed_hours:.1f} hours). Forcefully terminating the hung process to clear bottleneck."
                    )
                    try:
                        import signal

                        os.kill(lock_pid, signal.SIGKILL)
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Failed to kill hung process {lock_pid}: {e}")

                    try:
                        os.remove(lock_file)
                        logger.info(
                            "Hung process lock cleared. Proceeding with new pipeline."
                        )
                    except Exception:
                        pass
                else:
                    logger.warning(
                        f"Already running (Active process PID {lock_pid} detected, elapsed {elapsed_hours:.1f} hours). Terminating pipeline."
                    )
                    return
        else:
            try:
                os.remove(lock_file)
                logger.info("Removed corrupted lock file.")
            except Exception:
                pass

    try:
        # Create lock file
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"Starting Buddy Core Pipeline (Type: {report_type})...")

        if not is_us_trading_day():
            logger.info(
                "US Market is closed (Weekend/Holiday). Skipping pipeline execution today."
            )
            return

        # Start processing pipeline
        logger.info(f"[1/5] Pulling Sieve data for {report_type}...")
        pull_data_from_cloud(report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[2/5] Running Sorter...")
        run_sorter(report_type=report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[3/5] Running Extractor...")
        run_extractor()
        logger.info("-----------------------------------------------------")

        if report_type == "incremental":
            logger.info(
                "Incremental pipeline completed successfully up to Extractor. Exiting."
            )
            return

        logger.info("[4/5] Running CIO...")
        run_cio(report_type=report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[5/5] Running Formatter...")
        logger.info("Running Formatter for English report...")
        data_dir = os.path.join(project_root, "data")
        cio_file = os.path.join(
            data_dir,
            (
                f"premarket_report_{today_str}.txt"
                if report_type == "premarket"
                else f"final_report_{today_str}.txt"
            ),
        )
        out_file = os.path.join(
            data_dir,
            (
                f"alpha_signal_premarket_{today_str}.md"
                if report_type == "premarket"
                else f"alpha_signal_{today_str}.md"
            ),
        )
        success = run_formatter(cio_file, out_file)

        if success:
            if report_type == "premarket":
                logger.info(
                    "Premarket pipeline completed successfully. Cleaning up intermediate data files..."
                )
                _cleanup_data_files(data_dir)
                logger.info("Cleanup complete.")
            else:
                logger.info(
                    "Full pipeline completed successfully. Deferring cleanup until premarket run."
                )

            logger.info("Successfully completed Buddy Core Pipeline!")
        else:
            logger.warning("Translator did not return success. Skipping cleanup.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Cleanup (Remove lock file)
        if os.path.exists(lock_file):
            os.remove(lock_file)
            logger.info("Lock file removed.")


def _cleanup_data_files(data_dir: str):
    """Deletes all files in data_dir except alpha_signal_*.md"""
    if not os.path.exists(data_dir):
        return

    all_files = glob.glob(os.path.join(data_dir, "*"))
    for file_path in all_files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            # Keep the final alpha signal markdown files
            if not (filename.startswith("alpha_signal_") and filename.endswith(".md")):
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted {filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buddy Core Pipeline")
    parser.add_argument(
        "--type",
        choices=["full", "premarket", "incremental"],
        default="full",
        help="Type of report to generate",
    )
    args = parser.parse_args()
    run_all(report_type=args.type)
