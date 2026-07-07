import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
import shutil
from datetime import datetime
import pytz
import sys
import torch
import requests

# Add src and project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from extractor import run_extractor


class TestExtractorRealTimeTranslation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.us_tz = pytz.timezone("America/New_York")
        self.today_str = datetime.now(self.us_tz).strftime("%Y%m%d")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("extractor.SentenceTransformer")
    @patch("extractor.verify_exact_match")
    @patch("requests.Session.post")
    def test_run_extractor_real_time_translation_and_flush(
        self, mock_post, mock_verify, mock_sentence_transformer
    ):
        # 1. Mock SentenceTransformer
        mock_embedder = MagicMock()
        mock_sentence_transformer.return_value = mock_embedder
        mock_embedder.encode.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_embedder.similarity.side_effect = lambda a, b: [
            torch.tensor([0.0] * len(b))
        ]

        # 2. Mock verify_exact_match to always succeed
        mock_verify.return_value = True

        # 3. Mock requests.Session.post to return valid Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Extracted key performance metrics."}
        }
        mock_post.return_value = mock_response

        # 4. Create active tasks in a category file
        # We will create 2 normal articles and 2 SEC filings
        articles = [
            {
                "title": "SEC Filing A",
                "url": "https://sec.com/a",
                "content": "SEC content",
                "extraction_status": "sec_filing",
            },
            {
                "title": "Article 1",
                "url": "https://art1.com",
                "content": "Normal content 1",
            },
            {
                "title": "SEC Filing B",
                "url": "https://sec.com/b",
                "content": "SEC content",
                "extraction_status": "sec_filing",
            },
        ]

        category_file = os.path.join(
            self.test_dir, f"General_sorted_{self.today_str}.json"
        )
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(articles, f)

        # 5. Run extractor
        result = run_extractor(data_dir=self.test_dir)
        self.assertTrue(result)

        # 6. Assertions:
        # SEC filings should be bypassed and written directly into the cache.
        # Check translated_state JSON file exists
        cache_file = os.path.join(
            self.test_dir, f"translated_state_{self.today_str}.json"
        )
        self.assertTrue(os.path.exists(cache_file))

        with open(cache_file, "r", encoding="utf-8") as cf:
            cache_data = json.load(cf)

        # Assert SEC filings have empty translations
        self.assertIn("https://sec.com/a", cache_data)
        self.assertEqual(cache_data["https://sec.com/a"]["body"], "")
        self.assertIn("https://sec.com/b", cache_data)
        self.assertEqual(cache_data["https://sec.com/b"]["body"], "")

    @patch("extractor.SentenceTransformer")
    @patch("extractor.verify_exact_match")
    @patch("requests.Session.post")
    def test_run_extractor_consecutive_ollama_failures_aborts(
        self, mock_post, mock_verify, mock_sentence_transformer
    ):
        """Ollama failing consecutively 3 times should abort the extraction gracefully throwing RuntimeError."""
        # 1. Mock SentenceTransformer
        mock_embedder = MagicMock()
        mock_sentence_transformer.return_value = mock_embedder
        mock_embedder.encode.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_embedder.similarity.side_effect = lambda a, b: [
            torch.tensor([0.0] * len(b))
        ]

        # 2. Mock verify_exact_match to always succeed
        mock_verify.return_value = True

        # 3. Mock requests.Session.post to raise ConnectionError
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        # 4. Create 3 normal articles (enough to trigger 3 consecutive failures)
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://art{i}.com",
                "content": f"Normal content {i}",
            }
            for i in range(1, 5)
        ]

        category_file = os.path.join(
            self.test_dir, f"General_sorted_{self.today_str}.json"
        )
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(articles, f)

        # 5. Extraction should abort and throw RuntimeError
        with self.assertRaises(RuntimeError) as context:
            run_extractor(data_dir=self.test_dir)

        self.assertIn("Ollama server down", str(context.exception))


if __name__ == "__main__":
    unittest.main()
