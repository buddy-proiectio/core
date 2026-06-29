"""
The Translator Agent (English to Korean translation using custom backend engine)

Translates extracted articles using NLLB model via custom API endpoint at http://127.0.0.1:8000/translate,
manages translation caching to avoid redundant calls, and formats the final Korean reports.
"""

import os
import re
import sys
import json
import glob
import argparse
import requests
import pytz
import subprocess
import time
from datetime import datetime
from typing import Optional, List, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.shared_logger import setup_logger
from formatter import run_formatter

LOG_FILE = "logs/translator.log"
logger = setup_logger(LOG_FILE, __name__)

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


def translate_article(title: str, body: str) -> str:
    """
    Calls custom translation API at http://127.0.0.1:8000/translate.
    Translates title and body separately to preserve format and prevent truncation.
    """
    url = "http://127.0.0.1:8000/translate"

    # Translate title
    translated_title = ""
    if title.strip():
        payload = {"text": title.strip()}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            translated_title = resp.json().get("translated_text", "").strip()
        except Exception as e:
            logger.error(f"Custom translation API failed for title: {e}")
            translated_title = title.strip()

    # Translate body
    translated_body = ""
    if body.strip():
        payload = {"text": body.strip()}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            translated_body = resp.json().get("translated_text", "").strip()
        except Exception as e:
            logger.error(f"Custom translation API failed for body: {e}")
            translated_body = body.strip()

    if translated_title and translated_body:
        return f"{translated_title}\n{translated_body}"
    elif translated_title:
        return translated_title
    else:
        return translated_body


def translate_single_article_task(
    url: str, title: str, body: str
) -> tuple[str, str, str]:
    """Helper task to translate a single article."""
    logger.info(f"Translating article: {title[:50]}...")
    if re.search(r"\b(8-K|10-K|10-Q|SEC Filing)\b", title, re.IGNORECASE):
        # Do not translate standard SEC filing markers
        return url, title, ""
    else:
        translated_text = translate_article(title, body)
        if not translated_text:
            return url, "", ""

        parts = translated_text.split("\n", 1)
        translated_title = parts[0].strip()
        translated_body = parts[1].strip() if len(parts) > 1 else ""
        return url, translated_title, translated_body


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
    for block in blocks:
        url, title, body = parse_article_block(block)
        if not url:
            continue
        if limit_urls is not None and url not in limit_urls:
            continue
        if url in cache:
            continue
        to_translate.append((url, title, body))

    if not to_translate:
        logger.info("No new translations needed.")
        return

    logger.info(f"Translating {len(to_translate)} articles sequentially...")
    updated = False

    for url, title, body in to_translate:
        try:
            res_url, trans_title, trans_body = translate_single_article_task(
                url, title, body
            )
            if trans_title:
                cache[res_url] = {"title": trans_title, "body": trans_body}
                updated = True
            else:
                logger.warning(f"Translation failed or returned empty for URL: {url}")
        except Exception as e:
            logger.error(f"Error during translation for URL {url}: {e}")

    if updated:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated translation cache saved containing {len(cache)} items.")


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
            # Report header or other metadata
            ko_sections.append(sec_strip)

    ko_content = "\n\n".join(ko_sections)
    with open(ko_draft_file, "w", encoding="utf-8") as f:
        f.write(ko_content)
    logger.info(f"Generated Korean full report draft at {ko_draft_file}")
    return True


def is_server_healthy() -> bool:
    try:
        resp = requests.get("http://127.0.0.1:8000/", timeout=1)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("status") == "ok"
    except Exception:
        pass
    return False


def start_translation_server() -> Optional[subprocess.Popen]:
    """Starts the cortex uvicorn server in a subprocess."""
    if is_server_healthy():
        logger.info("Translation server is already running and healthy.")
        return None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cortex_dir = os.path.join(os.path.dirname(project_root), "cortex")
    python_bin = os.path.join(cortex_dir, ".venv", "bin", "python")

    logger.info("Starting translation server...")
    # Launch uvicorn as a subprocess without reload
    proc = subprocess.Popen(
        [
            python_bin,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=cortex_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the server to become healthy (max 30 seconds)
    start_time = time.time()
    while time.time() - start_time < 30:
        if is_server_healthy():
            logger.info(f"Translation server started successfully (pid: {proc.pid}).")
            return proc
        if proc.poll() is not None:
            logger.error("Translation server process exited prematurely.")
            break
        time.sleep(0.5)

    logger.error("Failed to start translation server within timeout.")
    try:
        proc.terminate()
    except Exception:
        pass
    return None


def stop_translation_server(proc: Optional[subprocess.Popen]) -> None:
    """Stops the uvicorn subprocess."""
    if proc is None:
        return
    logger.info(f"Stopping translation server (pid: {proc.pid})...")
    try:
        proc.terminate()
        proc.wait(timeout=10)
        logger.info("Translation server stopped successfully.")
    except subprocess.TimeoutExpired:
        logger.warning("Translation server terminate timed out, killing process...")
        proc.kill()
        proc.wait()
    except Exception as e:
        logger.error(f"Error stopping translation server: {e}")


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
    cache_file = os.path.join(data_dir, f"translated_state_{today_str}.json")

    logger.info(f"Running Translator (Type: {report_type}) for date {today_str}")

    # Spin up the translation server if not already running
    server_process = start_translation_server()
    try:
        if report_type == "incremental":
            translate_new_articles(state_file, cache_file)

        elif report_type == "premarket":
            # 1. Prioritize selected premarket articles
            en_report = os.path.join(data_dir, f"premarket_report_{today_str}.txt")

            # Fallback to latest premarket file if not exists
            if not os.path.exists(en_report):
                files = sorted(
                    glob.glob(os.path.join(data_dir, "premarket_report_*.txt"))
                )
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

            ko_output_dir = os.path.join(data_dir, "ko")
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

            ko_output_dir = os.path.join(data_dir, "ko")
            os.makedirs(ko_output_dir, exist_ok=True)
            ko_final_report = os.path.join(
                ko_output_dir, f"alpha_signal_{today_str}_ko.md"
            )

            run_formatter(ko_draft, ko_final_report, lang="ko")
            logger.info(
                f"Successfully generated final Korean full report at {ko_final_report}"
            )
    finally:
        # Guarantee the server is stopped if we launched it
        stop_translation_server(server_process)


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
