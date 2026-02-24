import logging
import sys


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        time_str = self.formatTime(record, "%H:%M:%S")
        level_name = f"{record.levelname:8}"

        # 기본적으로 초록색 (INFO 및 그 이하)
        color_code = "\x1b[32m"

        if record.levelno == logging.WARNING:
            color_code = "\x1b[33m"  # 노란색
        elif record.levelno >= logging.ERROR:
            color_code = "\x1b[31m"  # 빨간색

        reset_code = "\x1b[0m"

        return (
            f"{color_code}{time_str} - {level_name}{reset_code} | {record.getMessage()}"
        )


def setup_logger(log_file: str):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 기본 핸들러 초기화 (중복 방지)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 파일 출력용 평문 포매터
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # 콘솔 출력용 핸들러 (색상 포함)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())

    # 파일 출력용 핸들러
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
