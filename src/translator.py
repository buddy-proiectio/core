"""
Translator module for Buddy Core.

This module translates English markdown reports to publication-ready Korean reports.
It leverages Google AI Studio's Gemma model chain (defaulting to gemma-4-31b) in JSON mode,
maintains translation caches to minimize redundant translation costs, and localizes schedules.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import pytz
import requests
from formatter import run_formatter
from shared.env_utils import load_env_file
from shared.shared_logger import setup_logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Populate os.environ with local config prior to bootstrapping dependencies.
load_env_file()

LOG_FILE = "logs/translator.log"
logger = setup_logger(LOG_FILE, __name__)

# Model fallback chain configuration
# Can be overridden via TRANSLATOR_MODEL env var (e.g. TRANSLATOR_MODEL="gemma-4-31b,gemma-4-26b")
env_model = os.environ.get("TRANSLATOR_MODEL", "")
if env_model:
    MODEL_CHAIN = [m.strip() for m in env_model.split(",") if m.strip()]
else:
    MODEL_CHAIN = [
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemma-3-27b-it",
        "gemma-2-27b-it",
    ]

# In-memory registry to track which models do not support native responseSchema
# to avoid wasting API calls on subsequent batches.
SCHEMA_UNSUPPORTED_MODELS = set()

# In-memory registry to track which models do not support native thinkingConfig (thinkingBudget)
# to avoid wasting API calls on subsequent batches.
THINKING_CONFIG_UNSUPPORTED_MODELS = set()

# Category translation mapping: Translates English section headers to their official Korean equivalent
# used for publish-ready capital market reports in South Korea (e.g. "### General" -> "### 경제 일반").
CATEGORY_MAPPING = {
    "### General": "### 경제 일반",
    "### Bitcoin": "### 비트코인",
    "### Semiconductor": "### 반도체",
    "### AI/Robotics/EV": "### AI / 로봇 / EV",
    "### Power/Grid": "### 전력 / 인프라",
    "### Software": "### 소프트웨어",
    "### Aerospace": "### 우주 항공",
    "### Bio": "### 바이오",
    "### Consumer/Retail": "### 소비재 / 리테일",
    "### Others": "### 기타",
}


def normalize_category_header(header: str) -> str:
    """Normalize headers by removing whitespace and slashes for stable lookup."""
    return header.replace(" ", "").replace("/", "").strip()


# Precompute a normalized lookup mapping to resolve raw spaces and forward slashes safely.
NORM_MAPPING = {normalize_category_header(k): v for k, v in CATEGORY_MAPPING.items()}


def parse_article_block(block: str) -> tuple[str, str, str]:
    """
    Parses an article block into (url, title, body).
    Example: [Apple Sinks 6%...](https://...)\nBody text here...
    """
    match = re.match(r"^\[(.*?)\]\((.*?)\)(?:\s*<br\s*/?>)?(.*)", block, re.DOTALL)
    if match:
        title = match.group(1).strip()
        url = match.group(2).strip()
        body = match.group(3).strip()
        return url, title, body
    return "", "", ""


# Minimal fallback system prompt for the open-source release.
# Instructs the model to return JSON translations without complex capital-market noise filters.
DEFAULT_TRANSLATION_SYSTEM_PROMPT = """You are a professional financial translator specializing in translating US financial news and articles into high-quality, professional Korean.
You MUST output your response strictly as a JSON object containing a "translations" key mapped to an array of translated articles. Each item in the array must preserve the "url" key and contain the translated "title" and "body" keys."""

# Load the proprietary translation system prompt dynamically if it exists.
# Since config/ is gitignored, this allows local runs to keep proprietary translation IP.
TRANSLATION_SYSTEM_PROMPT = DEFAULT_TRANSLATION_SYSTEM_PROMPT

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_file_path = os.path.join(
        project_root, "config", "prompts", "translation_system_prompt.txt"
    )
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            custom_prompt = f.read().strip()
            if custom_prompt:
                TRANSLATION_SYSTEM_PROMPT = custom_prompt
except Exception:
    # Fail silently to maintain execution flow with the default prompt
    pass


def build_payload(user_prompt: str, model: str) -> dict:
    """Builds the API request payload dynamically based on model capabilities."""
    generation_config: Dict[str, Any] = {
        "temperature": 0.1,
        "responseMimeType": "application/json",
    }

    # Inject responseSchema if model is not registered as schema-unsupported
    if model not in SCHEMA_UNSUPPORTED_MODELS:
        generation_config["responseSchema"] = {
            "type": "OBJECT",
            "properties": {
                "translations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "url": {"type": "STRING"},
                            "title": {"type": "STRING"},
                            "body": {"type": "STRING"},
                        },
                        "required": ["url", "title", "body"],
                    },
                }
            },
            "required": ["translations"],
        }

    # Inject thinkingConfig with thinkingBudget=0 to disable thinking mode bottleneck
    # only if the model is not registered as unsupported
    if model not in THINKING_CONFIG_UNSUPPORTED_MODELS:
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": TRANSLATION_SYSTEM_PROMPT}]},
        "generationConfig": generation_config,
    }

    return payload


def call_gemini_translator_api(
    articles: List[Dict[str, str]],
    retries_per_model: int = 2,
    backoff_factor: int = 3,
) -> Dict[str, Dict[str, str]]:
    """
    Calls Google AI Studio's API to translate a batch of articles in JSON mode.
    Tries each model in MODEL_CHAIN sequentially. If a model fails, is rate-limited,
    or is not found, falls back to the next model in the chain.
    """
    # Fetch the Gemini API Key from environment variables.
    # If the key is not defined, we raise a ValueError to prevent empty/failing API calls.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not defined.")
    headers = {"Content-Type": "application/json"}
    user_prompt = json.dumps(articles, ensure_ascii=False)

    last_err = None
    for model in MODEL_CHAIN:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        for attempt in range(retries_per_model):
            try:
                # Build payload dynamically based on caches
                payload = build_payload(user_prompt, model)

                logger.info(
                    f"Calling API for translation batch of size {len(articles)} (Attempt {attempt + 1}/{retries_per_model}) using model {model}..."
                )
                resp = requests.post(url, json=payload, headers=headers, timeout=120)

                # Check for bad request due to unsupported configurations
                if resp.status_code == 400:
                    error_text = resp.text
                    retry_needed = False

                    if (
                        "responseSchema" in error_text
                        and model not in SCHEMA_UNSUPPORTED_MODELS
                    ):
                        logger.warning(
                            f"Model {model} returned 400 with responseSchema. Registering model as schema-unsupported."
                        )
                        SCHEMA_UNSUPPORTED_MODELS.add(model)
                        retry_needed = True

                    if (
                        "thinkingConfig" in error_text or "thinkingBudget" in error_text
                    ) and model not in THINKING_CONFIG_UNSUPPORTED_MODELS:
                        logger.warning(
                            f"Model {model} returned 400 with thinkingConfig. Registering model as thinkingConfig-unsupported."
                        )
                        THINKING_CONFIG_UNSUPPORTED_MODELS.add(model)
                        retry_needed = True

                    if retry_needed:
                        # Rebuild payload and retry immediately
                        retry_payload = build_payload(user_prompt, model)
                        resp = requests.post(
                            url, json=retry_payload, headers=headers, timeout=120
                        )

                # Check for Rate Limit (HTTP 429)
                if resp.status_code == 429:
                    sleep_time = backoff_factor * (attempt + 1)
                    logger.warning(
                        f"Rate limit (429) hit for model {model}. Sleeping for {sleep_time} seconds before retry..."
                    )
                    time.sleep(sleep_time)
                    continue

                # If it's a 404 (model not found) or 403 (permission / quota issue), skip this model immediately
                if resp.status_code in (403, 404):
                    logger.warning(
                        f"Model {model} returned HTTP {resp.status_code}. Skipping this model..."
                    )
                    break

                resp.raise_for_status()
                res_data = resp.json()

                candidates = res_data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"No candidates returned from model {model}")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"No parts found in candidate from model {model}")

                content_text = parts[0].get("text", "").strip()

                if content_text.startswith("```"):
                    content_text = re.sub(
                        r"^```(?:json)?\n", "", content_text, flags=re.IGNORECASE
                    )
                    content_text = re.sub(r"\n```$", "", content_text)
                content_text = content_text.strip()

                result = json.loads(content_text)
                translations_list = result.get("translations", [])

                mapped_results = {}
                for item in translations_list:
                    url_key = item.get("url")
                    if url_key:
                        mapped_results[url_key] = {
                            "title": item.get("title", "").strip(),
                            "body": item.get("body", "").strip(),
                        }
                return mapped_results

            except Exception as e:
                logger.error(f"Error calling translation API with model {model}: {e}")
                last_err = e
                if attempt < retries_per_model - 1:
                    sleep_time = backoff_factor * (attempt + 1)
                    logger.info(f"Retrying model {model} in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(
                        f"All attempts for model {model} failed. Falling back to next model..."
                    )

    if last_err:
        raise last_err
    else:
        raise ValueError(
            "Failed to translate batch: All fallback models in the chain failed."
        )


def translate_new_articles(
    state_file: str, cache_file: str, limit_urls: Optional[List[str]] = None
) -> None:
    """
    Load state_file and cache_file, translate any new articles that are NOT in the cache.
    If limit_urls is provided, we only translate those URLs.
    """
    if not os.path.exists(state_file):
        logger.warning(f"State file {state_file} does not exist. Skipping translation.")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load translation cache: {e}")

    # Gather all blocks from normal and sec outputs
    blocks = []
    for cat, items in state_data.get("category_normal_outputs", {}).items():
        blocks.extend(items)
    for cat, items in state_data.get("category_sec_outputs", {}).items():
        blocks.extend(items)

    # Gather items to translate
    to_translate = []
    updated = False

    for block in blocks:
        url, title, body = parse_article_block(block)
        if not url:
            continue
        if limit_urls is not None and url not in limit_urls:
            continue
        if url in cache:
            continue

        # Rule for SEC filings (8-K, 10-K, 10-Q, SEC Filing): do not translate body, cache immediately
        if re.search(r"\b(8-K|10-K|10-Q|SEC Filing)\b", title, re.IGNORECASE):
            logger.info(f"Skipping translation for SEC filing marker: {title[:50]}...")
            cache[url] = {"title": title.strip(), "body": ""}
            updated = True
            continue

        to_translate.append({"url": url, "title": title, "body": body})

    if not to_translate:
        if updated:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info("No new translations needed.")
        return

    logger.info(f"Translating {len(to_translate)} articles in batches...")

    batch_size = 10
    batches = [
        to_translate[i : i + batch_size]
        for i in range(0, len(to_translate), batch_size)
    ]

    for batch_idx, batch in enumerate(batches):
        try:
            # Enforce 2-second cooldown delay between calls to respect RPM 15
            if batch_idx > 0:
                logger.info(
                    "Sleeping 2 seconds to avoid exceeding RPM 15 rate limit..."
                )
                time.sleep(2.0)

            translations = call_gemini_translator_api(batch)

            batch_updated = False
            for article in batch:
                url = article["url"]
                if url in translations:
                    cache[url] = translations[url]
                    batch_updated = True
                    updated = True
                else:
                    logger.warning(f"Translation output missing for url: {url}")

            if batch_updated:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"Incremental translation cache saved for batch {batch_idx + 1}/{len(batches)}."
                )

        except Exception as e:
            logger.error(f"Error during translation of batch {batch_idx + 1}: {e}")

    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Final translation cache saved containing {len(cache)} items.")


def extract_urls_from_report(report_file: str) -> List[str]:
    """Extract all article URLs from a formatted report."""
    if not os.path.exists(report_file):
        return []
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    return re.findall(r"\[.*?\]\((https?://.*?)\)", content)


def generate_korean_premarket_draft(
    en_report_file: str, ko_draft_file: str, cache_file: str
) -> bool:
    """Generates the Korean premarket report draft from the English report and cache."""
    if not os.path.exists(en_report_file):
        logger.error(f"English premarket report {en_report_file} does not exist.")
        return False

    with open(en_report_file, "r", encoding="utf-8") as f:
        content = f.read()

    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    header_match = re.match(
        r"^##\s+(.*?)\s+Premarket\s*\n*(.*)", content, re.DOTALL | re.IGNORECASE
    )
    if not header_match:
        logger.error("Failed to parse English premarket report header.")
        return False

    date_str = header_match.group(1).strip()
    body_content = header_match.group(2).strip()

    blocks = re.split(r"(?=\[.*?\]\(https?://.*?\))", body_content)
    ko_blocks = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        url, title, body = parse_article_block(block)
        if url and url in cache:
            ko_title = cache[url]["title"]
            ko_body = cache[url]["body"]
            if ko_body:
                ko_blocks.append(f"[{ko_title}]({url})\n{ko_body}")
            else:
                ko_blocks.append(f"[{ko_title}]({url})")
        else:
            ko_blocks.append(block)

    ko_content = f"## {date_str} Premarket\n\n" + "\n\n".join(ko_blocks)
    with open(ko_draft_file, "w", encoding="utf-8") as f:
        f.write(ko_content)
    logger.info(f"Generated Korean premarket draft at {ko_draft_file}")
    return True


def generate_korean_full_draft(
    en_report_file: str, ko_draft_file: str, cache_file: str
) -> bool:
    """Generates the Korean full report draft from the English report and cache."""
    if not os.path.exists(en_report_file):
        logger.error(f"English full report {en_report_file} does not exist.")
        return False

    with open(en_report_file, "r", encoding="utf-8") as f:
        content = f.read()

    cache: Dict[str, Dict[str, str]] = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # 1. Strip ### Daily Point
    dp_match = re.search(r"###\s+Daily\s+Point", content, re.IGNORECASE)
    if dp_match:
        start_dp = dp_match.start()
        next_sec = re.search(
            r"(\n###\s+Weekly\s+Schedule|\n###\s+주간\s*일정)",
            content[dp_match.end() :],
            re.IGNORECASE,
        )
        if next_sec:
            end_dp = dp_match.end() + next_sec.start()
            content = content[:start_dp].rstrip() + "\n\n" + content[end_dp:].lstrip()
        else:
            logger.warning("Could not find next section after Daily Point to delete.")

    # 2. Map Category Headers and replace article blocks
    sections = re.split(r"(?=\n###\s+)", content)
    ko_sections = []

    for sec in sections:
        sec_strip = sec.strip()
        if not sec_strip:
            continue

        lines = sec_strip.split("\n")
        header = lines[0].strip()

        norm_header = normalize_category_header(header)
        if norm_header in NORM_MAPPING:
            ko_header = NORM_MAPPING[norm_header]
            body_text = "\n".join(lines[1:]).strip()
            blocks = re.split(r"(?=\[.*?\]\(https?://.*?\))", body_text)

            ko_blocks = []
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                url, title, body = parse_article_block(block)
                if url and url in cache:
                    ko_title = cache[url]["title"]
                    ko_body = cache[url]["body"]
                    if ko_body:
                        ko_blocks.append(f"[{ko_title}]({url})\n{ko_body}")
                    else:
                        ko_blocks.append(f"[{ko_title}]({url})")
                else:
                    ko_blocks.append(block)

            ko_sec = f"{ko_header}\n\n" + "\n\n".join(ko_blocks)
            ko_sections.append(ko_sec)
        elif norm_header == normalize_category_header(
            "### Weekly Schedule"
        ) or norm_header == normalize_category_header("### 주간 일정"):
            ko_sections.append("### 주간 일정\n" + "\n".join(lines[1:]))
        else:
            ko_sections.append(sec_strip)

    ko_content = "\n\n".join(ko_sections)
    with open(ko_draft_file, "w", encoding="utf-8") as f:
        f.write(ko_content)
    logger.info(f"Generated Korean full report draft at {ko_draft_file}")
    return True


def run_translator(report_type: str = "full") -> None:
    """
    Main entry point for translation pipeline.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")

    us_tz = pytz.timezone("America/New_York")
    today_str = datetime.now(us_tz).strftime("%Y%m%d")

    # Check if a custom json file is explicitly passed to capture testing
    custom_date = None
    for arg in sys.argv[1:]:
        m = re.search(r"(\d{8})", arg)
        if m:
            custom_date = m.group(1)
            break

    if custom_date:
        today_str = custom_date

    state_file = os.path.join(data_dir, f"extracted_state_{today_str}.json")

    if report_type == "premarket":
        cache_file = os.path.join(data_dir, "translated_state_pre.json")
    else:
        cache_file = os.path.join(data_dir, f"translated_state_{today_str}.json")

        # Merge premarket translation cache if it exists from previous premarket runs
        pre_cache_file = os.path.join(data_dir, "translated_state_pre.json")
        if os.path.exists(pre_cache_file):
            try:
                with open(pre_cache_file, "r", encoding="utf-8") as f:
                    pre_cache = json.load(f)

                standard_cache = {}
                if os.path.exists(cache_file):
                    with open(cache_file, "r", encoding="utf-8") as f:
                        standard_cache = json.load(f)

                merged_cache = {**pre_cache, **standard_cache}
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(merged_cache, f, ensure_ascii=False, indent=2)

                logger.info(
                    f"Merged {len(pre_cache)} translation keys from translated_state_pre.json into today's cache."
                )
            except Exception as e:
                logger.error(f"Failed to merge translated_state_pre.json: {e}")

    logger.info(f"Running Translator (Type: {report_type}) for date {today_str}")

    if report_type == "incremental":
        translate_new_articles(state_file, cache_file)

    elif report_type == "premarket":
        # 1. Prioritize selected premarket articles
        en_report = os.path.join(data_dir, f"premarket_report_{today_str}.txt")

        # Fallback to latest premarket file if not exists
        if not os.path.exists(en_report):
            files = sorted(glob.glob(os.path.join(data_dir, "premarket_report_*.txt")))
            if files:
                en_report = files[-1]

        premarket_urls = extract_urls_from_report(en_report)
        logger.info(
            f"Premarket report selected {len(premarket_urls)} articles for translation."
        )

        # Translate only selected premarket articles first
        translate_new_articles(state_file, cache_file, limit_urls=premarket_urls)

        # Generate KST premarket draft and format it immediately
        ko_draft = os.path.join(data_dir, f"premarket_report_ko_{today_str}.txt")
        generate_korean_premarket_draft(en_report, ko_draft, cache_file)

        ko_output_dir = os.path.join(data_dir, "premarket")
        os.makedirs(ko_output_dir, exist_ok=True)
        ko_final_report = os.path.join(
            ko_output_dir, f"alpha_signal_premarket_{today_str}_ko.md"
        )

        run_formatter(ko_draft, ko_final_report, lang="ko")
        logger.info(
            f"Successfully generated final Korean premarket report at {ko_final_report}"
        )

        # 2. In the background, translate all remaining premarket articles
        logger.info("Proceeding to translate all remaining articles in state...")
        translate_new_articles(state_file, cache_file)

    elif report_type == "full":
        # 1. Translate all articles to complete the cache
        translate_new_articles(state_file, cache_file)

        # 2. Read full English report
        en_report = os.path.join(data_dir, f"final_report_{today_str}.txt")
        if not os.path.exists(en_report):
            files = sorted(glob.glob(os.path.join(data_dir, "final_report_*.txt")))
            if files:
                en_report = files[-1]

        ko_draft = os.path.join(data_dir, f"final_report_ko_{today_str}.txt")
        generate_korean_full_draft(en_report, ko_draft, cache_file)

        ko_output_dir = os.path.join(data_dir, "report")
        os.makedirs(ko_output_dir, exist_ok=True)
        ko_final_report = os.path.join(ko_output_dir, f"alpha_signal_{today_str}_ko.md")

        run_formatter(ko_draft, ko_final_report, lang="ko")
        logger.info(
            f"Successfully generated final Korean full report at {ko_final_report}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buddy Core Translation Pipeline")
    parser.add_argument(
        "--type",
        choices=["full", "premarket", "incremental"],
        default="full",
        help="Type of report to translate",
    )
    args = parser.parse_args()
    run_translator(report_type=args.type)
