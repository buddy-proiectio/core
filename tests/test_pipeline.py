import unittest
from unittest.mock import patch
from datetime import datetime
import pytz
import os
import sys

# Ensure project root and src are in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import run_all


class TestPipelineWeekendBypass(unittest.TestCase):
    @patch("src.datetime")
    @patch("src.is_us_trading_day")
    @patch("src.pull_data_from_cloud")
    @patch("src.run_sorter")
    @patch("src.run_extractor")
    @patch("src.run_translator")
    @patch("src.os.path.exists")
    @patch("src.os.remove")
    @patch("src.open")
    @patch("src.os.makedirs")
    def test_run_all_non_incremental_on_weekend_skips(
        self,
        mock_makedirs,
        mock_open,
        mock_remove,
        mock_exists,
        mock_translator,
        mock_extractor,
        mock_sorter,
        mock_pull,
        mock_is_trading_day,
        mock_datetime,
    ):
        # Setup: non-trading day (weekend)
        mock_is_trading_day.return_value = False
        mock_exists.return_value = False  # lock file doesn't exist

        # Setup mock datetime to return a time past 06:00
        mock_now = datetime(
            2026, 7, 4, 12, 0, 0, tzinfo=pytz.timezone("America/New_York")
        )
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.strptime

        # Test full report
        run_all(report_type="full")

        # Verify pull_data_from_cloud was NOT called because we skipped on weekend
        mock_pull.assert_not_called()

    @patch("src.datetime")
    @patch("src.is_us_trading_day")
    @patch("src.pull_data_from_cloud")
    @patch("src.run_sorter")
    @patch("src.run_extractor")
    @patch("src.run_translator")
    @patch("src.os.path.exists")
    @patch("src.os.remove")
    @patch("src.open")
    @patch("src.os.makedirs")
    def test_run_all_incremental_on_weekend_bypasses_skip(
        self,
        mock_makedirs,
        mock_open,
        mock_remove,
        mock_exists,
        mock_translator,
        mock_extractor,
        mock_sorter,
        mock_pull,
        mock_is_trading_day,
        mock_datetime,
    ):
        # Setup: non-trading day (weekend)
        mock_is_trading_day.return_value = False
        mock_exists.return_value = False  # lock file doesn't exist

        # Setup mock datetime to return a time past 06:00
        mock_now = datetime(
            2026, 7, 4, 12, 0, 0, tzinfo=pytz.timezone("America/New_York")
        )
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.strptime

        mock_pull.return_value = None

        # Test incremental report
        run_all(report_type="incremental")

        # Verify pull_data_from_cloud WAS called because incremental bypasses the weekend skip
        mock_pull.assert_called_once_with("incremental")
        mock_sorter.assert_called_once_with("incremental")
        mock_extractor.assert_called_once_with(report_type="incremental")
        mock_translator.assert_called_once_with("incremental")


if __name__ == "__main__":
    unittest.main()
