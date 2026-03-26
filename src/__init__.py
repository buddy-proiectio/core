"""
Core processing modules for Buddy Core
"""

from sorter import run_sorter
from extractor import run_extractor
from cio import run_cio
from translator import run_translator
import os
import subprocess
from datetime import datetime
import glob
import logging
import pytz
import holidays

from shared_logger import setup_logger

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


def pull_data_from_cloud():
    """
    Pull sieve's data from oracle cloud storage
    """
    today = datetime.now().strftime("%Y%m%d")

    """
    Change the following variables if you are not using the same environment as mine
    """
    ORACLE_IP_ADDRESS = "159.13.60.28"
    ORACLE_SSH_KEY = "/Users/taehoonkwon__/.ssh/oracle-cloud-ssh.key"
    remote_file = f"/home/ubuntu/data/daily_news_{today}.json"

    local_dir = "/Users/taehoonkwon__/workspaces/buddy-core/data"
    local_file = f"{local_dir}/daily_news_{today}.json"

    os.makedirs(local_dir, exist_ok=True)

    scp_command = f"scp -i {ORACLE_SSH_KEY} -o StrictHostKeyChecking=no ubuntu@{ORACLE_IP_ADDRESS}:{remote_file} {local_file}"

    try:
        logger.info(f"Pulling data from oracle cloud storage... ({remote_file})")

        subprocess.run(
            scp_command, shell=True, check=True, capture_output=True, text=True
        )

        if os.path.exists(local_file):
            logger.info(
                f"Successfully pulled data from oracle cloud storage: {local_file}"
            )
        else:
            logger.error(
                f"Not found {local_file} after pulling from oracle cloud storage"
            )
            raise FileNotFoundError(
                f"Not found {local_file} after pulling from oracle cloud storage"
            )

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to pull data from oracle cloud storage: {e}")
        raise


def run_all():
    """
    Run the entire data processing pipeline sequentially:
    Sorter -> Extractor -> CIO -> Translator

    After translator finishes successfully, cleans up all files in the /data directory
    EXCEPT for the final alpha_signal_*.md file.
    """
    logger.info("Starting Buddy Core Pipeline...")

    if not is_us_trading_day():
        logger.info(
            "US Market is closed (Weekend/Holiday). Skipping pipeline execution today."
        )
        return

    try:
        logger.info(f"[0/4] Pulling Sieve data...")
        pull_data_from_cloud()
        logger.info("-----------------------------------------------------")

        logger.info("[1/4] Running Sorter...")
        run_sorter()
        logger.info("-----------------------------------------------------")

        logger.info("[2/4] Running Extractor...")
        run_extractor()
        logger.info("-----------------------------------------------------")

        logger.info("[3/4] Running CIO...")
        run_cio()
        logger.info("-----------------------------------------------------")

        logger.info("[4/4] Running Translator...")
        success = run_translator()
        logger.info("-----------------------------------------------------")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

        if success:
            logger.info(
                "Pipeline completed successfully. Cleaning up intermediate data files..."
            )
            _cleanup_data_files(data_dir)
            logger.info("Cleanup complete.")
            logger.info("Successfully completed Buddy Core Pipeline!")
        else:
            logger.warning("Translator did not return success. Skipping cleanup.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


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
    run_all()
