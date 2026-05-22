"""
The Chief Translator (English to Korean)

Translates English text to professional Korean using litellm and ollama/translategemma.
Implements a robust retry logic for transient errors.
"""

import os
import re
import sys
import glob
import logging
import argparse

LOG_FILE = "logs/translator.log"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root to sys.path to allow importing from 'shared'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.shared_logger import setup_logger

logger = setup_logger(LOG_FILE, __name__)

# Suppress noisy INFO logs from external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)

import translators as ts


def chunk_text(text: str, limit: int = 4000) -> list[str]:
    """Simple text chunker to ensure no piece exceeds the limit."""
    chunks = []
    while len(text) > limit:
        split_index = text.rfind(" ", 0, limit)
        if split_index == -1:
            split_index = limit

        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()

    if text:
        chunks.append(text)
    return chunks


def make_polite(text: str) -> str:
    """
    Post-process Korean translation to enforce polite/formal endings (~입니다, ~합니다).
    DeepL API currently does not support the 'formality' parameter for Korean.
    """
    if not text:
        return text

    # Negative lookahead: only replace if NOT followed by another Korean character.
    # This safely prevents modifying connectors like ~한다고, ~한다면, ~했다며, etc.
    suffix_pattern = r"(?![가-힣ㄱ-ㅎㅏ-ㅣ])"

    # 1. Noun exceptions and explicit special cases
    explicit_replacements = [
        (r"수치다", "수치입니다"),
        (r"결과다", "결과입니다"),
        (r"목표다", "목표입니다"),
        (r"규모다", "규모입니다"),
        (r"추세다", "추세입니다"),
        (r"상태다", "상태입니다"),
        (r"이유다", "이유입니다"),
        (r"예상치다", "예상치입니다"),
        (r"전망치다", "전망치입니다"),
        (r"최고치다", "최고치입니다"),
        (r"최저치다", "최저치입니다"),
        (r"기록이다", "기록입니다"),
        (r"상황이다", "상황입니다"),
        (r"수준이다", "수준입니다"),
        (r"예정이다", "예정입니다"),
        (r"것이다", "것입니다"),
        (r"전망이다", "전망입니다"),
        (r"예상이다", "예상입니다"),
        (r"예측이다", "예측입니다"),
        (r"중이다", "중입니다"),
        (r"때문이다", "때문입니다"),
        (r"이다", "입니다"),
        (r"아니다", "아닙니다"),
    ]
    for old, new in explicit_replacements:
        text = re.sub(old + suffix_pattern, new, text)

    # 2. Convert ~는다 to ~습니다 (for consonant-ending present tense verbs like 받는다)
    text = re.sub(r"([가-힣])는다" + suffix_pattern, r"\1습니다", text)

    # 3. Present tense, Adjectives, and common verbs
    replacements = [
        (r"한다", "합니다"),
        (r"하다", "합니다"),
        (r"된다", "됩니다"),
        (r"되다", "됩니다"),
        (r"있다", "있습니다"),
        (r"없다", "없습니다"),
        (r"않다", "않습니다"),
        (r"크다", "큽니다"),
        (r"많다", "많습니다"),
        (r"적다", "적습니다"),
        (r"높다", "높습니다"),
        (r"낮다", "낮습니다"),
        (r"같다", "같습니다"),
        (r"다르다", "다릅니다"),
        (r"작다", "작습니다"),
        (r"가깝다", "가깝습니다"),
        (r"멀다", "멉니다"),
        (r"빠르다", "빠릅니다"),
        (r"느리다", "느립니다"),
        (r"어렵다", "어렵습니다"),
        (r"쉽다", "쉽습니다"),
        (r"어려워진다", "어려워집니다"),
        (r"쉬워진다", "쉬워집니다"),
        (r"좋다", "좋습니다"),
        (r"나쁘다", "나쁩니다"),
        (r"강하다", "강합니다"),
        (r"약하다", "약합니다"),
        (r"새롭다", "새롭습니다"),
        (r"필요하다", "필요합니다"),
        (r"중요하다", "중요합니다"),
        (r"가능하다", "가능합니다"),
        (r"불가능하다", "불가능합니다"),
        (r"심하다", "심합니다"),
        (r"비슷하다", "비슷합니다"),
        (r"충분하다", "충분합니다"),
        (r"다양하다", "다양합니다"),
        (r"유사하다", "유사합니다"),
        (r"겠다", "겠습니다"),
        (r"진다", "집니다"),
        (r"시킨다", "시킵니다"),
        (r"나온다", "나옵니다"),
        (r"보인다", "보입니다"),
        (r"준다", "줍니다"),
        (r"나타난다", "나타납니다"),
        (r"간다", "갑니다"),
        (r"온다", "옵니다"),
        (r"늘어난다", "늘어납니다"),
        (r"가져온다", "가져옵니다"),
        (r"줄어든다", "줄어듭니다"),
        (r"떨어진다", "떨어집니다"),
        (r"오른다", "오릅니다"),
        (r"이어진다", "이어집니다"),
        (r"커진다", "커집니다"),
        (r"작아진다", "작아집니다"),
        (r"강조한다", "강조합니다"),
        (r"성장한다", "성장합니다"),
        (r"하락한다", "하락합니다"),
        (r"증가한다", "증가합니다"),
        (r"감소한다", "감소합니다"),
        (r"상승한다", "상승합니다"),
    ]
    for old, new in replacements:
        text = re.sub(old + suffix_pattern, new, text)

    # 4. Programmatic replacer for Past Tense (any Syllable with ㅆ 받침 + 다)
    # This safely handles 했, 었, 았, 였, 났, 겼, 혔, 샀, 컸, 줬, 갔, 왔, 등등
    def ssang_sios_replacer(match):
        char = match.group(1)
        if "가" <= char <= "힣":
            char_code = ord(char) - 0xAC00
            if char_code % 28 == 20:  # 20 is the index for ㅆ 받침
                return char + "습니다"
        return match.group(0)

    text = re.sub(r"([가-힣])다" + suffix_pattern, ssang_sios_replacer, text)

    return text


def contains_english(t: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", t))


def contains_korean(t: str) -> bool:
    return bool(re.search(r"[가-힣]", t))


def translate_text_with_retry(text: str) -> str:
    import time
    import random

    proxies = {"http": "socks5://127.0.0.1:9050", "https": "socks5://127.0.0.1:9050"}

    for attempt in range(15):
        method = attempt % 3
        try:
            if method == 0:
                # Google with Tor
                translated = ts.translate_text(
                    text,
                    from_language="en",
                    to_language="ko",
                    translator="google",
                    proxies=proxies,
                    timeout=10,
                )
            elif method == 1:
                # Google direct (no proxy)
                translated = ts.translate_text(
                    text,
                    from_language="en",
                    to_language="ko",
                    translator="google",
                    timeout=10,
                )
            else:
                # Bing direct (no proxy)
                translated = ts.translate_text(
                    text,
                    from_language="en",
                    to_language="ko",
                    translator="bing",
                    timeout=10,
                )

            translated_str = str(translated).strip()

            # If successful, check if it actually translated to Korean
            if translated_str and contains_korean(translated_str):
                return make_polite(translated_str)

        except Exception as e:
            logger.warning(f"Translation attempt {attempt+1} failed: {e}")

        # Exponential backoff with jitter
        sleep_time = 0.5 * (2 ** (attempt % 3)) + random.uniform(0.1, 0.3)
        time.sleep(sleep_time)

    # Absolute fallback: return original text but log error
    logger.error(f"All 15 translation attempts failed for: '{text}'")
    return text


def translate_text(text: str) -> str:
    if not text.strip():
        return text

    # If the text has no english remaining, return it directly!
    if not contains_english(text):
        return text

    return translate_text_with_retry(text)


def process_line(line: str) -> str:
    """
    Process a single line of markdown text, handling structural symbols and links constraints.
    """
    # 1. Blank lines preserved as blank lines without calling LLM
    if not line.strip():
        return line

    # Strip newline at the right to reconstruct it safely later
    original_endswith_newline = line.endswith("\n")
    line_content = line.rstrip("\n")

    # Check if the line begins with any of our structural symbols (ignoring leading whitespace)
    stripped_line = line_content.lstrip()

    # --- Hard-Tuning (1:1 Matching) ---

    # 1) Exact matches (Headers)
    exact_mappings = {
        "### Daily Point": "### Daily Point",
        "**Topline Signals**": "**Topline Signals**",
        "### Weekly Schedule": "### 주간 일정",
        "### General": "### 경제 일반",
        "### Bitcoin": "### 비트코인",
        "### Semiconductor": "### 반도체",
        "### AI": "### AI",
        "### Bio": "### 바이오",
        "### Aerospace": "### 우주항공",
        "### Software": "### 소프트웨어",
        "### Others": "### 기타",
    }
    if stripped_line in exact_mappings:
        return line_content.replace(stripped_line, exact_mappings[stripped_line]) + (
            "\n" if original_endswith_newline else ""
        )

    # 1.5) Markdown Tables (Do not translate)
    if (
        stripped_line.startswith("|")
        and stripped_line.endswith("|")
        and "|" in stripped_line[1:-1]
    ):
        return line_content + ("\n" if original_endswith_newline else "")

    # 2) Daily Point Indicators (Do not translate Dow Jones, Nasdaq, S&P 500, Bitcoin)
    if stripped_line.startswith("_ "):
        if any(
            indicator in stripped_line
            for indicator in [
                "Dow Jones",
                "S&P 500",
                "Nasdaq",
                "Bitcoin",
            ]
        ):
            return line_content + ("\n" if original_endswith_newline else "")

    # 3) Dates: Report Header (e.g., ## 20260224) - Keep as-is for the formatter to handle
    m_report = re.match(r"^##\s+(\d{4})(\d{2})(\d{2})$", stripped_line)
    if m_report:
        return line_content + ("\n" if original_endswith_newline else "")

    # 2. Check for structural symbols
    structural_symbols = ["## ", "### ", "* ", "★ ", "_ "]
    prefix = ""
    target_text = line_content

    for sym in structural_symbols:
        if stripped_line.startswith(sym):
            # Calculate where the symbol ends to split the line
            prefix_len = len(line_content) - len(stripped_line) + len(sym)
            prefix = line_content[:prefix_len]
            target_text = line_content[prefix_len:]
            break

    # If the rest of the text is empty, no need to query LLM
    if not target_text.strip():
        # Re-attach the newline if it existed
        return prefix + ("\n" if original_endswith_newline else "")

    # 3. Check for Markdown Links
    pattern = r"\[(.*?)\]\((.*?)\)"

    # If there are links, we process them carefully without destroying markdown
    if re.search(pattern, target_text):
        parts = []
        last_idx = 0
        for match in re.finditer(pattern, target_text):
            # Text before the link
            before = target_text[last_idx : match.start()]
            if before.strip():
                parts.append(translate_text(before))
            else:
                parts.append(before)

            # Extract title and url
            title = match.group(1)
            url = match.group(2)

            # Send ONLY the title to translate_text (skip for SEC filings)
            if re.search(r"\b(8-K|10-K|10-Q|SEC Filing)\b", title, re.IGNORECASE):
                translated_title = title
            else:
                translated_title = translate_text(title)

            # Reassemble strictly using Python (without <br /> formatting)
            parts.append(f"[{translated_title}]({url})")

            last_idx = match.end()

        # Text after the last link
        after = target_text[last_idx:]
        if after.strip():
            parts.append(translate_text(after))
        else:
            parts.append(after)

        translated_text = "".join(parts)
    else:
        # 4. For all other plain text
        translated_text = translate_text(target_text)

    # Reattach symbol and newline
    final_line = prefix + translated_text

    final_line += "\n" if original_endswith_newline else ""
    return final_line


def run_translator(report_type: str = "full"):
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if report_type == "premarket":
        search_pattern = os.path.join(data_dir, "premarket_report_*.txt")
    else:
        search_pattern = os.path.join(data_dir, "final_report_*.txt")

    files = glob.glob(search_pattern)

    if not files:
        logger.error(f"No input files matching {search_pattern} found.")
        sys.exit(1)

    # Process the most recent file
    files.sort(reverse=True)
    input_file = files[0]
    filename = os.path.basename(input_file)

    if report_type == "premarket":
        date_part = filename.replace("premarket_report_", "").replace(".txt", "")
        output_filename = f"premarket_report_ko_draft_{date_part}.txt"
    else:
        date_part = filename.replace("final_report_", "").replace(".txt", "")
        output_filename = f"final_report_ko_draft_{date_part}.txt"
    output_file = os.path.join(data_dir, output_filename)

    logger.info(f"Starting translation process...")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output Draft: {output_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []
    in_weekly_schedule = False

    for i, line in enumerate(lines):
        try:
            stripped = line.strip()

            # If we hit another section header, exit weekly schedule skip mode
            if stripped.startswith("### ") and stripped != "### Weekly Schedule":
                in_weekly_schedule = False

            if in_weekly_schedule:
                # Bypass translation entirely for weekly schedule contents
                translated_line = line
            else:
                translated_line = process_line(line)

            # If we just hit the Weekly Schedule header, enter skip mode
            if stripped == "### Weekly Schedule":
                in_weekly_schedule = True

        except KeyboardInterrupt:
            logger.warning("Translation process interrupted. Exiting gracefully.")
            return False
        except Exception as e:
            logger.error(
                f"Failed to translate line {i+1}: '{line.strip()}'. Keeping original. Error: {e}"
            )
            translated_line = line

        translated_lines.append(translated_line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(translated_lines)

    logger.info(f"Translation complete. Saved draft to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translator Script")
    parser.add_argument(
        "--type",
        choices=["full", "premarket"],
        default="full",
        help="Type of report to translate",
    )
    args = parser.parse_args()
    run_translator(report_type=args.type)
