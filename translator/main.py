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

LOG_FILE = "translator.log"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_logger import setup_logger

setup_logger(LOG_FILE)
logger = logging.getLogger(__name__)

from litellm import completion
import litellm

litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.turn_off_message_logging = True

# Additionally suppress LiteLLM's internal logger
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def translate_text(text: str) -> str:
    if not text.strip():
        return text

    system_prompt = "Translate the following English (en) financial/macroeconomic text into natural, highly professional Korean (ko-KR). Do not add any extra commentary or conversational filler. Output ONLY the translated text."

    max_retries = 3
    delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = completion(
                model="ollama/translategemma",
                api_base="http://localhost:11434",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            translated = response.choices[0].message.content.strip()
            return translated
        except Exception as e:
            logger.warning(
                f"Translation failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Raising exception.")
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

    # 3) Dates: Report Header (e.g., ## 20260224 Report -> ## 2026년 2월 24일 보고서)
    m_report = re.match(r"^##\s+(\d{4})(\d{2})(\d{2})\s+Report$", stripped_line)
    if m_report:
        year, month, day = m_report.groups()
        replaced = f"## {year}년 {int(month)}월 {int(day)}일 보고서"
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
    tags_mapping = {"[Macro]": "[매크로]", "[Earnings]": "[실적]"}
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


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # We are looking for files matching final_newsletter_YYYYMMDD.txt
    search_pattern = os.path.join(workspace_dir, "final_newsletter_*.txt")
    files = glob.glob(search_pattern)

    if not files:
        logger.error("No input files matching final_newsletter_YYYYMMDD.txt found.")
        sys.exit(1)

    # Process the most recent file
    files.sort(reverse=True)
    input_file = files[0]

    filename = os.path.basename(input_file)
    date_part = filename.replace("final_newsletter_", "").replace(".txt", "")

    output_filename = f"alpha_signal_{date_part}.md"
    output_file = os.path.join(workspace_dir, output_filename)

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
                ("_ ", "* ", "★ ", "[Macro]", "[Earnings]", "[매크로]", "[실적]")
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


if __name__ == "__main__":
    main()
