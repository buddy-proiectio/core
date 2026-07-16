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
    @patch("src.get_next_us_trading_day")
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
        mock_get_next,
        mock_is_trading_day,
        mock_datetime,
    ):
        # Setup: non-trading day (weekend)
        mock_is_trading_day.return_value = False
        mock_get_next.return_value = datetime(
            2026, 7, 6, 12, 0, 0, tzinfo=pytz.timezone("America/New_York")
        )
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
    @patch("src.get_next_us_trading_day")
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
        mock_get_next,
        mock_is_trading_day,
        mock_datetime,
    ):
        # Setup: non-trading day (weekend)
        mock_is_trading_day.return_value = False
        mock_get_next.return_value = datetime(
            2026, 7, 6, 12, 0, 0, tzinfo=pytz.timezone("America/New_York")
        )
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
        mock_pull.assert_called_once_with("incremental", target_date="20260706")
        mock_sorter.assert_called_once_with("incremental", target_date="20260706")
        mock_extractor.assert_called_once_with(
            report_type="incremental", target_date="20260706"
        )
        mock_translator.assert_called_once_with("incremental", target_date="20260706")


class TestPullDataFromCloudPaths(unittest.TestCase):
    """Verify that pull_data_from_cloud uses target_date for file path construction."""

    @patch("src.subprocess.run")
    def test_pull_data_uses_target_date_for_incremental(
        self, mock_subprocess_run
    ):
        """When target_date is provided (weekend scenario), SCP should target that date's file."""
        from src import pull_data_from_cloud

        # Simulate a successful SCP pull with a valid JSON file
        mock_subprocess_run.return_value = None  # check=True won't raise

        target_date = "20260706"  # Monday (next trading day from Saturday July 4)

        # Create a temporary data dir and file to simulate successful pull
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        test_file = os.path.join(data_dir, f"daily_news_{target_date}.json")

        try:
            # Write a valid JSON file so the function finds it after SCP
            with open(test_file, "w") as f:
                import json

                json.dump({"articles": [], "market_map": {}, "weekly_schedule": {}}, f)

            pull_data_from_cloud("incremental", target_date=target_date)

            # Verify SCP command was called with the target date in the remote path
            mock_subprocess_run.assert_called_once()
            scp_cmd = mock_subprocess_run.call_args[0][0]
            self.assertIn(f"daily_news_{target_date}.json", scp_cmd)
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)

    @patch("src.subprocess.run")
    def test_pull_data_uses_target_date_for_premarket(
        self, mock_subprocess_run
    ):
        """Premarket pulls should also use target_date for the file name."""
        from src import pull_data_from_cloud

        mock_subprocess_run.return_value = None

        target_date = "20260706"

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        test_file = os.path.join(data_dir, f"premarket_news_{target_date}.json")

        try:
            with open(test_file, "w") as f:
                import json

                json.dump({"articles": [], "market_map": {}, "weekly_schedule": {}}, f)

            pull_data_from_cloud("premarket", target_date=target_date)

            mock_subprocess_run.assert_called_once()
            scp_cmd = mock_subprocess_run.call_args[0][0]
            self.assertIn(f"premarket_news_{target_date}.json", scp_cmd)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == "__main__":
    unittest.main()
