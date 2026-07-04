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

# Add src and project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import run_extractor


class TestExtractorRealTimeTranslation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.us_tz = pytz.timezone("America/New_York")
        self.today_str = datetime.now(self.us_tz).strftime("%Y%m%d")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.extractor.SentenceTransformer")
    @patch("src.extractor.verify_exact_match")
    @patch("src.extractor.call_gemini_translator_api")
    @patch("requests.Session.post")
    def test_run_extractor_real_time_translation_and_flush(
        self, mock_post, mock_translator_api, mock_verify, mock_sentence_transformer
    ):
        # mock_post -> requests.Session.post
        # mock_translator_api -> call_gemini_translator_api
        # mock_verify -> verify_exact_match
        # mock_sentence_transformer -> SentenceTransformer

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

        # 4. Mock Gemini translator API response
        # It should return a dict mapping url -> {"title": ko_title, "body": ko_body}
        def mock_translate_side_effect(articles):
            res = {}
            for art in articles:
                res[art["url"]] = {
                    "title": f"Translated Title of {art['title']}",
                    "body": f"Translated Body of {art['body']}",
                }
            return res

        mock_translator_api.side_effect = mock_translate_side_effect

        # 5. Create active tasks in a category file
        # We will create 6 normal articles and 2 SEC filings
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
                "title": "Article 2",
                "url": "https://art2.com",
                "content": "Normal content 2",
            },
            {
                "title": "Article 3",
                "url": "https://art3.com",
                "content": "Normal content 3",
            },
            {
                "title": "Article 4",
                "url": "https://art4.com",
                "content": "Normal content 4",
            },
            {
                "title": "SEC Filing B",
                "url": "https://sec.com/b",
                "content": "SEC content",
                "extraction_status": "sec_filing",
            },
            {
                "title": "Article 5",
                "url": "https://art5.com",
                "content": "Normal content 5",
            },
            {
                "title": "Article 6",
                "url": "https://art6.com",
                "content": "Normal content 6",
            },
        ]

        category_file = os.path.join(
            self.test_dir, f"General_sorted_{self.today_str}.json"
        )
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(articles, f)

        # 6. Run extractor
        result = run_extractor(data_dir=self.test_dir)
        self.assertTrue(result)

        # 7. Assertions:
        # SEC filings should bypassed and written directly into the cache.
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

        # Assert normal articles are translated and in the cache
        for i in range(1, 7):
            url = f"https://art{i}.com"
            self.assertIn(url, cache_data)
            self.assertEqual(
                cache_data[url]["title"], f"Translated Title of Article {i}"
            )

        # Assert mock_translator_api was called twice (once for batch size 4, once for flushing remaining 2)
        self.assertEqual(mock_translator_api.call_count, 2)

        # Check details of the first call (batch of 4)
        first_call_args = mock_translator_api.call_args_list[0][0][0]
        self.assertEqual(len(first_call_args), 4)
        self.assertEqual(first_call_args[0]["url"], "https://art1.com")
        self.assertEqual(first_call_args[3]["url"], "https://art4.com")

        # Check details of the second call (flush of 2)
        second_call_args = mock_translator_api.call_args_list[1][0][0]
        self.assertEqual(len(second_call_args), 2)
        self.assertEqual(second_call_args[0]["url"], "https://art5.com")
        self.assertEqual(second_call_args[1]["url"], "https://art6.com")

    @patch("src.extractor.SentenceTransformer")
    @patch("src.extractor.verify_exact_match")
    @patch("src.extractor.call_gemini_translator_api")
    @patch("requests.Session.post")
    def test_run_extractor_real_time_translation_failure_raises_runtime_error(
        self, mock_post, mock_translator_api, mock_verify, mock_sentence_transformer
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

        # 4. Mock Gemini translator API to fail
        mock_translator_api.side_effect = Exception("Translation API down")

        # 5. Create 4 normal articles
        articles = [
            {
                "title": "Article 1",
                "url": "https://art1.com",
                "content": "Normal content 1",
            },
            {
                "title": "Article 2",
                "url": "https://art2.com",
                "content": "Normal content 2",
            },
            {
                "title": "Article 3",
                "url": "https://art3.com",
                "content": "Normal content 3",
            },
            {
                "title": "Article 4",
                "url": "https://art4.com",
                "content": "Normal content 4",
            },
        ]

        category_file = os.path.join(
            self.test_dir, f"General_sorted_{self.today_str}.json"
        )
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(articles, f)

        # 6. Verify that it raises RuntimeError when translation fails
        with self.assertRaises(RuntimeError) as context:
            run_extractor(data_dir=self.test_dir)

        self.assertIn("Real-time translation pipeline failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
