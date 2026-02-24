"""
Core processing modules for Buddy Core
"""

from sorter import run_sorter
from extractor import run_extractor
from cio import run_cio
from translator import run_translator
import os
import glob
import logging

logger = logging.getLogger(__name__)


def run_all():
    """
    Run the entire data processing pipeline sequentially:
    Sorter -> Extractor -> CIO -> Translator

    After translator finishes successfully, cleans up all files in the /data directory
    EXCEPT for the final alpha_signal_*.md file.
    """
    logger.info("Starting Buddy Core Pipeline...")

    try:
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
