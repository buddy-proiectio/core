import logging
import sys


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        time_str = self.formatTime(record, "%H:%M:%S")
        level_name = f"{record.levelname:8}"

        # default Green (INFO, etc.)
        color_code = "\x1b[32m"

        if record.levelno == logging.WARNING:
            color_code = "\x1b[33m"  # Yellow
        elif record.levelno >= logging.ERROR:
            color_code = "\x1b[31m"  # Red

        reset_code = "\x1b[0m"

        return (
            f"{color_code}{time_str} - {level_name}{reset_code} | {record.getMessage()}"
        )


import os


def setup_logger(log_file: str):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Reset default handler
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File output formatter
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # Console output handler (with color)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())

    # File output handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
