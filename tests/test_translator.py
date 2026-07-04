import unittest
from unittest.mock import patch
import os
import json
import tempfile
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from translator import translate_new_articles, translate_missing_report_articles, TranslationError


class TestTranslatorFailFast(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "state.json")
        self.cache_file = os.path.join(self.test_dir, "cache.json")
        self.report_file = os.path.join(self.test_dir, "report.txt")

        # Create a valid state file with some articles
        state_data = {
            "category_normal_outputs": {
                "General": [
                    "[Apple Sinks 6%](https://apple.com)\nApple shares fell on Friday."
                ]
            }
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f)

        # Create a valid report file with missing translations
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write("[Tesla Rallies 5%](https://tesla.com)\nTesla stock surged.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("translator.call_gemini_translator_api")
    def test_translate_new_articles_raises_runtime_error(self, mock_api):
        mock_api.side_effect = Exception("API connection timeout")

        with self.assertRaises(TranslationError) as context:
            translate_new_articles(self.state_file, self.cache_file)

        self.assertIn(
            "Translation pipeline failed during batch 1", str(context.exception)
        )

    @patch("translator.call_gemini_translator_api")
    def test_translate_missing_report_articles_raises_runtime_error(self, mock_api):
        mock_api.side_effect = Exception("Quota exceeded")

        with self.assertRaises(TranslationError) as context:
            translate_missing_report_articles(self.report_file, self.cache_file)

        self.assertIn("Retry translation failed during batch 1", str(context.exception))


if __name__ == "__main__":
    unittest.main()
