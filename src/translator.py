"""
The Chief Translator (English to Korean)

Translates English text to professional Korean using litellm and ollama/translategemma.
Implements a robust retry logic for transient errors.
"""

import os
import re
import sys
import time
import glob
import logging

LOG_FILE = "logs/translator.log"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared_logger import setup_logger

setup_logger(LOG_FILE)
logger = logging.getLogger(__name__)

# Suppress noisy INFO logs from external libraries
logging.getLogger("deepl").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from deep_translator import GoogleTranslator
import deepl

DEEPL_API_KEY = "41415273-1167-e96a-d51f-7c6d52c06ac2:fx"
deepl_translator = deepl.Translator(DEEPL_API_KEY)


def make_polite(text: str) -> str:
    """
    Post-process Korean translation to enforce polite/formal endings (~입니다, ~합니다).
    DeepL API currently does not support the 'formality' parameter for Korean.
    """
    if not text:
        return text
        
    # Past Tense
    text = re.sub(r'([었았였했])다\.', r'\1습니다.', text)
    text = re.sub(r'([었았였했])다([\s\n])', r'\1습니다\2', text)
    text = re.sub(r'([었았였했])다$', r'\1습니다', text)
    
    # Present Tense & Adjectives
    replacements = [
        (r'한다\.', '합니다.'),
        (r'하다\.', '합니다.'),
        (r'된다\.', '됩니다.'),
        (r'이다\.', '입니다.'),
        (r'있다\.', '있습니다.'),
        (r'없다\.', '없습니다.'),
        (r'않다\.', '않습니다.'),
        (r'크다\.', '큽니다.'),
        (r'많다\.', '많습니다.'),
        (r'적다\.', '적습니다.'),
        (r'높다\.', '높습니다.'),
        (r'낮다\.', '낮습니다.'),
        (r'같다\.', '같습니다.'),
        (r'겠다\.', '겠습니다.'),
        (r'진다\.', '집니다.'),
        (r'시킨다\.', '시킵니다.'),
        (r'나온다\.', '나옵니다.'),
        (r'보인다\.', '보입니다.'),
        (r'준다\.', '줍니다.'),
        (r'받는다\.', '받습니다.'),
        (r'간다\.', '갑니다.'),
        (r'온다\.', '옵니다.'),
        (r'증가했다\.', '증가했습니다.'),
        (r'감소했다\.', '감소했습니다.'),
        (r'상승했다\.', '상승했습니다.'),
        (r'하락했다\.', '하락했습니다.')
    ]
    
    for old, new in replacements:
        text = re.sub(old, new, text)
        # Catch items without periods
        old_no_dot = old.replace(r'\.', r'$')
        new_no_dot = new.replace('.', '')
        text = re.sub(old_no_dot, new_no_dot, text, flags=re.MULTILINE)
        
    return text

def translate_text(text: str) -> str:
    if not text.strip():
        return text

    max_retries = 3
    delay = 2  # retry delay in seconds

    # First attempt with DeepL
    for attempt in range(max_retries):
        try:
            result = deepl_translator.translate_text(text, target_lang="KO")
            return make_polite(result.text)
        except deepl.QuotaExceededException:
            logger.warning(
                "DeepL quota exceeded. Falling back to Google Translator immediately."
            )
            break
        except Exception as e:
            logger.warning(
                f"DeepL translation failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error(
                    "Max retries reached for DeepL. Falling back to Google Translator."
                )
                break

    # Fallback to Google Translator if DeepL failed or quota exceeded
    try:
        # To avoid IP block from Google Translator
        time.sleep(0.1)
        google_translator = GoogleTranslator(source="en", target="ko")
        translated = google_translator.translate(text)
        return make_polite(translated)
    except Exception as e:
        logger.error(f"Fallback Google translation failed: {e}")
        raise e


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

    # 2.5) Earnings Schedule (Do not translate)
    if stripped_line.startswith("★ [Earnings]"):
        return line_content.replace("[Earnings]", "[실적]") + (
            "\n" if original_endswith_newline else ""
        )

    # 3) Dates: Report Header (e.g., ## 20260224 -> ## 2026년 2월 24일 Alpha Signal
    m_report = re.match(r"^##\s+(\d{4})(\d{2})(\d{2})$", stripped_line)
    if m_report:
        year, month, day = m_report.groups()
        replaced = f"## {year}년 {int(month)}월 {int(day)}일 Alpha Signal"
        return line_content.replace(stripped_line, replaced) + (
            "\n" if original_endswith_newline else ""
        )

    # 4) Dates: Weekly Schedule (e.g., Feb 24 (Tue) -> 2월 24일 (화))
    date_pattern = r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)\s*\((Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\)$"
    m_date = re.match(date_pattern, stripped_line)
    if m_date:
        month_str, day_str, dow_str = m_date.groups()
        months_kr = {
            "Jan": "1월",
            "Feb": "2월",
            "Mar": "3월",
            "Apr": "4월",
            "May": "5월",
            "Jun": "6월",
            "Jul": "7월",
            "Aug": "8월",
            "Sep": "9월",
            "Oct": "10월",
            "Nov": "11월",
            "Dec": "12월",
            "January": "1월",
            "February": "2월",
            "March": "3월",
            "April": "4월",
            "June": "6월",
            "July": "7월",
            "August": "8월",
            "September": "9월",
            "October": "10월",
            "November": "11월",
            "December": "12월",
        }
        days_kr = {
            "Mon": "월",
            "Tue": "화",
            "Wed": "수",
            "Thu": "목",
            "Fri": "금",
            "Sat": "토",
            "Sun": "일",
            "Monday": "월요일",
            "Tuesday": "화요일",
            "Wednesday": "수요일",
            "Thursday": "목요일",
            "Friday": "금요일",
            "Saturday": "토요일",
            "Sunday": "일요일",
        }
        replaced = f"{months_kr.get(month_str, month_str)} {int(day_str)}일 ({days_kr.get(dow_str, dow_str)})"
        return line_content.replace(stripped_line, replaced) + (
            "\n" if original_endswith_newline else ""
        )

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

    # 5) Partial string replacements: [Macro] -> [매크로], [Earnings] -> [실적]
    tags_mapping = {
        "[Macro]": "[매크로]",
        "[Earnings]": "[실적]",
    }
    for en_tag, kr_tag in tags_mapping.items():
        target_text_lstrip = target_text.lstrip()
        if target_text_lstrip.startswith(en_tag):
            after_tag = target_text_lstrip[len(en_tag) :]
            prefix += kr_tag + (" " if after_tag.startswith(" ") else "")
            target_text = after_tag.lstrip()
            break

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

            # Send ONLY the title to translate_text
            translated_title = translate_text(title)

            # Reassemble strictly using Python
            parts.append(f"[{translated_title}]({url})\\")

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


def run_translator():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # We are looking for files matching final_report_YYYYMMDD.txt
    search_pattern = os.path.join(data_dir, "final_report_*.txt")
    files = glob.glob(search_pattern)

    if not files:
        logger.error("No input files matching final_report_YYYYMMDD.txt found.")
        sys.exit(1)

    # Process the most recent file
    files.sort(reverse=True)
    input_file = files[0]

    filename = os.path.basename(input_file)
    date_part = filename.replace("final_report_", "").replace(".txt", "")

    output_filename = f"alpha_signal_{date_part}.md"
    output_file = os.path.join(data_dir, output_filename)

    logger.info(f"Starting translation process...")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []

    for i, line in enumerate(lines):
        try:
            translated_line = process_line(line)

            # Post-process to smartly add a markdown line break ('\')
            # if this line is part of a list/schedule and the NEXT line is not empty.
            stripped_original = line.lstrip()
            date_pattern = r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)\s*\("

            is_target_line = stripped_original.startswith(
                ("_ ", "* ", "★ ", "[Macro]", "[Earnings]")
            ) or re.match(date_pattern, stripped_original)

            if is_target_line:
                if i + 1 < len(lines) and lines[i + 1].strip():
                    if translated_line.endswith("\n"):
                        translated_line = translated_line[:-1] + "\\\n"
                    else:
                        translated_line += "\\"

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

    logger.info(f"Translation complete. Successfully saved to {output_file}")
    return True
