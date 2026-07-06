import unittest
from unittest.mock import patch
import os
import json
import tempfile
import shutil
import sys
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from translator import (
    translate_new_articles,
    translate_missing_report_articles,
    TranslationError,
)


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


class TestTranslatorSplitAndRetry(unittest.TestCase):
    @patch("translator.requests.post")
    @patch("translator.time.sleep")
    def test_split_and_retry_on_timeout(self, mock_sleep, mock_post):
        articles = [
            {"url": "https://apple.com", "title": "Apple Title", "body": "Apple Body"},
            {"url": "https://tesla.com", "title": "Tesla Title", "body": "Tesla Body"},
        ]

        def side_effect(url, **kwargs):
            json_payload = kwargs.get("json")
            assert json_payload is not None
            payload_text = json_payload["contents"][0]["parts"][0]["text"]
            payload_data = json.loads(payload_text)

            if len(payload_data) == 2:
                raise requests.exceptions.Timeout("Read timeout")
            elif len(payload_data) == 1:
                art = payload_data[0]
                url_placeholder = art["url"]
                resp = unittest.mock.MagicMock()
                resp.status_code = 200

                translated_art = {
                    "url": url_placeholder,
                    "title": f"Translated {art['title']}",
                    "body": f"Translated {art['body']}",
                }
                resp.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {"translations": [translated_art]}
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
                return resp
            else:
                raise ValueError("Unexpected batch size in mock")

        mock_post.side_effect = side_effect

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            from translator import call_gemini_translator_api, COOLDOWN_SLEEP_SECONDS

            results = call_gemini_translator_api(
                articles, retries_per_model=1, backoff_factor=1
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results["https://apple.com"]["title"], "Translated Apple Title"
        )
        self.assertEqual(
            results["https://tesla.com"]["title"], "Translated Tesla Title"
        )

        # Verify cooldown sleep was called between batch runs
        mock_sleep.assert_any_call(COOLDOWN_SLEEP_SECONDS)

    @patch("translator.requests.post")
    @patch("translator.time.sleep")
    def test_split_and_retry_on_json_error(self, mock_sleep, mock_post):
        articles = [
            {"url": "https://apple.com", "title": "Apple Title", "body": "Apple Body"},
            {"url": "https://tesla.com", "title": "Tesla Title", "body": "Tesla Body"},
        ]

        def side_effect(url, **kwargs):
            json_payload = kwargs.get("json")
            assert json_payload is not None
            payload_text = json_payload["contents"][0]["parts"][0]["text"]
            payload_data = json.loads(payload_text)

            resp = unittest.mock.MagicMock()
            resp.status_code = 200

            if len(payload_data) == 2:
                resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "invalid_json{"}]}}]
                }
            elif len(payload_data) == 1:
                art = payload_data[0]
                url_placeholder = art["url"]
                translated_art = {
                    "url": url_placeholder,
                    "title": f"Translated {art['title']}",
                    "body": f"Translated {art['body']}",
                }
                resp.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {"translations": [translated_art]}
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
            return resp

        mock_post.side_effect = side_effect

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            from translator import call_gemini_translator_api, COOLDOWN_SLEEP_SECONDS

            results = call_gemini_translator_api(
                articles, retries_per_model=1, backoff_factor=1
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results["https://apple.com"]["title"], "Translated Apple Title"
        )
        self.assertEqual(
            results["https://tesla.com"]["title"], "Translated Tesla Title"
        )

        # Verify cooldown sleep was called between batch runs
        mock_sleep.assert_any_call(COOLDOWN_SLEEP_SECONDS)

    @patch("translator.requests.post")
    @patch("translator.time.sleep")
    def test_no_split_on_single_article(self, mock_sleep, mock_post):
        articles = [
            {"url": "https://apple.com", "title": "Apple Title", "body": "Apple Body"}
        ]

        mock_post.side_effect = requests.exceptions.Timeout("Read timeout")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            from translator import call_gemini_translator_api

            with self.assertRaises(TranslationError):
                call_gemini_translator_api(
                    articles, retries_per_model=1, backoff_factor=1
                )


if __name__ == "__main__":
    unittest.main()
