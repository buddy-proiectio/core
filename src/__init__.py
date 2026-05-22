"""
Core processing modules for Buddy Core
"""

from sorter import run_sorter
from extractor import run_extractor
from cio import run_cio
from translator import run_translator
from formatter import run_formatter
import sys
import os
import subprocess
import time
import argparse
from datetime import datetime
import glob
import pytz
import holidays

# Add project root to sys.path to allow importing from 'shared'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    ORACLE_SSH_KEY = "/Users/taehoonkwon__/.ssh/oracle-cloud-ssh.key"

    if report_type == "premarket":
        remote_file = f"/home/ubuntu/data/premarket_news_{today}.json"
        local_dir = "/Users/taehoonkwon__/workspaces/buddy-core/data"
        local_file = f"{local_dir}/premarket_news_{today}.json"
    else:
        # Both 'full' and 'incremental' use the daily_news file
        remote_file = f"/home/ubuntu/data/daily_news_{today}.json"
        local_dir = "/Users/taehoonkwon__/workspaces/buddy-core/data"
        local_file = f"{local_dir}/daily_news_{today}.json"

    os.makedirs(local_dir, exist_ok=True)
    scp_command = f"scp -i {ORACLE_SSH_KEY} -o StrictHostKeyChecking=no ubuntu@{ORACLE_IP_ADDRESS}:{remote_file} {local_file}"

    max_retries = 5
    retry_delay = 60  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Pulling data from oracle cloud storage... (Attempt {attempt}/{max_retries})"
            )

            result = subprocess.run(
                scp_command, shell=True, check=True, capture_output=True, text=True
            )

            if os.path.exists(local_file):
                logger.info(f"Successfully pulled data: {local_file}")
                return  # Success, exit function
            else:
                raise FileNotFoundError(f"File not found after scp: {local_file}")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
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
    Sorter -> Extractor -> CIO -> Translator

    After translator finishes successfully, cleans up all files in the /data directory
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

    # 2. Duplicate Execution Prevention (Lock File)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_file = os.path.join(project_root, "logs", "buddy.lock")

    if os.path.exists(lock_file):
        logger.warning("Already running (buddy.lock exists). Terminating pipeline.")
        return

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
        logger.info(f"[0/5] Pulling Sieve data for {report_type}...")
        pull_data_from_cloud(report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[1/5] Running Sorter...")
        run_sorter(report_type=report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[2/5] Running Extractor...")
        run_extractor()
        logger.info("-----------------------------------------------------")

        if report_type == "incremental":
            logger.info("Incremental extraction complete. Skipping other processes.")
            return

        logger.info("[3/5] Running CIO...")
        run_cio(report_type=report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[4/5] Running Translator...")
        ko_draft_file = run_translator(report_type=report_type)
        logger.info("-----------------------------------------------------")

        logger.info("[5/5] Running Formatter...")
        logger.info("Running Formatter for English report...")
        data_dir = os.path.join(project_root, "data")
        en_cio_file = os.path.join(
            data_dir,
            (
                f"premarket_report_{today_str}.txt"
                if report_type == "premarket"
                else f"final_report_{today_str}.txt"
            ),
        )
        en_out_file = os.path.join(
            data_dir,
            (
                f"alpha_signal_premarket_{today_str}_en.md"
                if report_type == "premarket"
                else f"alpha_signal_{today_str}_en.md"
            ),
        )
        success_en = run_formatter(en_cio_file, en_out_file, lang="en")

        success_ko = False
        if ko_draft_file and os.path.exists(ko_draft_file):
            logger.info("Running Formatter for Korean report...")
            ko_out_file = os.path.join(
                data_dir,
                (
                    f"alpha_signal_premarket_{today_str}.md"
                    if report_type == "premarket"
                    else f"alpha_signal_{today_str}.md"
                ),
            )
            success_ko = run_formatter(ko_draft_file, ko_out_file, lang="ko")
        else:
            logger.error(
                "Korean draft translation file was not generated or does not exist."
            )

        success = success_en and success_ko

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

            # Automatically push to GitHub
            push_to_github(data_dir, report_type)

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


def push_to_github(data_dir: str, report_type: str = "full"):
    """
    Adds, commits, and pushes the final alpha signal reports to GitHub.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 1. Git Add (Specific to the alpha signal files)
        # Using shell=True to allow glob expansion
        subprocess.run(
            "git add data/alpha_signal_*.md",
            shell=True,
            cwd=project_root,
            check=True,
        )

        # 2. Git Commit
        commit_message = f"docs: add daily {report_type} alpha signal report: {datetime.now().strftime('%Y-%m-%d')}"
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", commit_message],
            cwd=project_root,
            check=True,
        )

        # 3. Git Push
        subprocess.run(["git", "push", "origin", "main"], cwd=project_root, check=True)

        logger.info("Successfully pushed final reports to GitHub.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
    except Exception as e:
        logger.error(f"An error occurred during git push: {e}")


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
