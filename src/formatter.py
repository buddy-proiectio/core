"""
Formatter module for Buddy Core.

This module formats and structures daily markdown reports. It builds timezone-aligned
weekly schedules (resolving the midnight timezone distortion bug), standardizes earnings
call labels, and enforces strict markdown compliance (line breaks, escaped math variables).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Optional
import pytz
from shared.shared_logger import setup_logger
from shared.time_utils import parse_utc_time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FILE = "logs/formatter.log"

logger = setup_logger(LOG_FILE, __name__)


def build_english_weekly_schedule(
    events: list, base_date_str: Optional[str] = None
) -> str:
    WEEKDAYS_EN = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    MONTHS_EN = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    TICKER_TO_COMPANY_NAME = {}
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        targets_file = os.path.join(project_root, "shared", "market_map_targets.json")
        if os.path.exists(targets_file):
            with open(targets_file, "r", encoding="utf-8") as f:
                targets_data = json.load(f)
                for item in targets_data:
                    symbol = item.get("Symbol")
                    company_name = item.get("Company Name")
                    if symbol and company_name:
                        TICKER_TO_COMPANY_NAME[symbol.upper()] = company_name
        else:
            logger.warning(f"market_map_targets.json not found at {targets_file}")
    except Exception as e:
        logger.error(f"Failed to load market_map_targets.json: {e}")

    # Fallback mappings for aliases
    if "GOOG" not in TICKER_TO_COMPANY_NAME and "GOOGL" in TICKER_TO_COMPANY_NAME:
        TICKER_TO_COMPANY_NAME["GOOG"] = TICKER_TO_COMPANY_NAME["GOOGL"]

    ny_tz = pytz.timezone("America/New_York")

    # Determine base date
    if base_date_str:
        try:
            base_date = datetime.strptime(base_date_str, "%Y%m%d").date()
        except ValueError:
            base_date = datetime.now(ny_tz).date()
    else:
        base_date = datetime.now(ny_tz).date()

    # Generate NY EST-aligned 7 consecutive days

    target_dates = [base_date + timedelta(days=i) for i in range(7)]
    grouped = {d: [] for d in target_dates}

    for event in events:
        utc_time_str = event.get("utc_time")
        if not utc_time_str:
            continue

        utc_dt = parse_utc_time(utc_time_str)

        # TIMEZONE MIDNIGHT BUG FIX
        if utc_dt.hour == 0 and utc_dt.minute == 0 and utc_dt.second == 0:
            local_date = utc_dt.date()
        else:
            local_dt = utc_dt.astimezone(ny_tz)
            local_date = local_dt.date()

        importance = event.get("importance", "medium")
        name = event.get("name", "").strip()

        if importance == "holiday":
            evt_str = f"(Holiday) {name}"
        elif importance == "earnings":
            # Map ticker NVDA -> NVIDIA
            ticker = name.split()[0].upper()
            company_name = TICKER_TO_COMPANY_NAME.get(ticker, ticker)
            evt_str = f"{company_name} Earnings Call"
        else:
            evt_str = name

        # Only add if it falls within the 7-day range
        if local_date in grouped:
            grouped[local_date].append(evt_str)

    # Construct the EN weekly schedule block
    lines = []
    for d in target_dates:
        weekday_en = WEEKDAYS_EN[d.weekday()]
        month_en = MONTHS_EN[d.month - 1]
        header = f"{d.day} {month_en} ({weekday_en})"
        lines.append(header)
        for evt in grouped[d]:
            lines.append(evt)
        lines.append("")

    return "\n".join(lines).strip()


def build_korean_weekly_schedule(
    events: list, base_date_str: Optional[str] = None
) -> str:
    # WEEKDAYS_KO mapping: Chronological mapping to full Korean weekday names for publishing aesthetics.
    WEEKDAYS_KO = [
        "월요일",
        "화요일",
        "수요일",
        "목요일",
        "금요일",
        "토요일",
        "일요일",
    ]
    # CURRENCY_TO_COUNTRY mapping: Translate ISO currencies to Korean localized country/region terms
    # to provide clear macroeconomic context to South Korean readers.
    CURRENCY_TO_COUNTRY = {
        "USD": "미국",
        "KRW": "한국",
        "JPY": "일본",
        "EUR": "EU",
        "GBP": "영국",
        "CAD": "캐나다",
        "AUD": "호주",
        "CNY": "중국",
    }

    # TICKER_TO_KOREAN_NAME mapping: Maps stock symbols to their Korean corporate titles
    # using targets from market_map_targets.json (e.g. NVDA -> 엔비디아).
    TICKER_TO_KOREAN_NAME = {}
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        targets_file = os.path.join(project_root, "shared", "market_map_targets.json")
        if os.path.exists(targets_file):
            with open(targets_file, "r", encoding="utf-8") as f:
                targets_data = json.load(f)
                for item in targets_data:
                    symbol = item.get("Symbol")
                    korean_name = item.get("Korean Name")
                    if symbol and korean_name:
                        TICKER_TO_KOREAN_NAME[symbol.upper()] = korean_name
        else:
            logger.warning(f"market_map_targets.json not found at {targets_file}")
    except Exception as e:
        logger.error(f"Failed to load market_map_targets.json: {e}")

    # Fallback mappings for aliases
    if "GOOG" not in TICKER_TO_KOREAN_NAME and "GOOGL" in TICKER_TO_KOREAN_NAME:
        TICKER_TO_KOREAN_NAME["GOOG"] = TICKER_TO_KOREAN_NAME["GOOGL"]

    ny_tz = pytz.timezone("America/New_York")

    # Determine base date
    if base_date_str:
        try:
            base_date = datetime.strptime(base_date_str, "%Y%m%d").date()
        except ValueError:
            base_date = datetime.now(ny_tz).date()
    else:
        base_date = datetime.now(ny_tz).date()

    # Generate NY EST-aligned 7 consecutive days

    target_dates = [base_date + timedelta(days=i) for i in range(7)]
    grouped = {d: [] for d in target_dates}

    for event in events:
        utc_time_str = event.get("utc_time")
        if not utc_time_str:
            continue

        utc_dt = parse_utc_time(utc_time_str)

        # TIMEZONE MIDNIGHT BUG FIX
        if utc_dt.hour == 0 and utc_dt.minute == 0 and utc_dt.second == 0:
            local_date = utc_dt.date()
        else:
            local_dt = utc_dt.astimezone(ny_tz)
            local_date = local_dt.date()

        currency = event.get("currency", "USD")
        importance = event.get("importance", "medium")
        name = event.get("name", "").strip()
        korean_name = event.get("korean_name", "").strip()

        country = CURRENCY_TO_COUNTRY.get(currency, currency)

        # Format calendar descriptions for Korean-language newsletters.
        # - "미국 증시 휴장" indicates US Market Holiday.
        # - "실적발표" indicates Corporate Earnings Call.
        if importance == "holiday":
            evt_str = (
                f"미국 증시 휴장 - {korean_name}"
                if korean_name
                else f"미국 증시 휴장 - {name}"
            )
        elif importance == "earnings":
            # Extract stock symbol and replace with mapped Korean corporate name (e.g. MSFT -> 마이크로소프트)
            ticker = name.split()[0].upper()
            company_name = TICKER_TO_KOREAN_NAME.get(ticker, ticker)
            evt_str = f"{company_name} 실적발표"
        else:
            clean_ko_name = korean_name if korean_name else name
            if country and clean_ko_name.startswith(country):
                clean_ko_name = clean_ko_name[len(country) :].strip()
                clean_ko_name = clean_ko_name.lstrip("-/ ").strip()
            evt_str = f"({country}) {clean_ko_name}"

        # Only add if it falls within the 7-day range
        if local_date in grouped:
            grouped[local_date].append(evt_str)

    # Construct the KO weekly schedule block
    lines = []
    for d in target_dates:
        weekday_ko = WEEKDAYS_KO[d.weekday()]
        header = f"{d.month}월 {d.day}일 ({weekday_ko})"
        lines.append(header)
        for evt in grouped[d]:
            lines.append(evt)
        lines.append("")

    return "\n".join(lines).strip()


def replace_weekly_schedule(content: str, schedule_str: str) -> str:
    # Find header
    header_pattern = r"(###\s+(?:Weekly\s+Schedule|주간\s*일정))"
    match = re.search(header_pattern, content, re.IGNORECASE)
    if not match:
        return content

    start_idx = match.end()

    # Find the next section header starting with "### "
    next_match = re.search(r"(\n###\s+)", content[start_idx:])
    if next_match:
        end_idx = start_idx + next_match.start()
        rest = content[end_idx:]
    else:
        rest = ""

    # Replace the schedule section content
    new_content = content[:start_idx] + "\n" + schedule_str + "\n\n" + rest
    return new_content


def format_content(
    content: str, lang: str = "en", weekly_schedule_data: Optional[list] = None
) -> str:
    """
    Applies markdown formatting rules to the report content.
    """
    if not content:
        return ""

    # 0. Reconstruct and replace the Weekly Schedule section with timezone-aligned schedule
    date_str = None
    m_date_raw = re.search(r"##\s*(\d{8})", content)

    if m_date_raw:
        date_str = m_date_raw.group(1)

    if date_str and not weekly_schedule_data:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        news_file = os.path.join(data_dir, f"daily_news_{date_str}.json")
        if os.path.exists(news_file):
            try:
                with open(news_file, "r", encoding="utf-8") as f:
                    news_data = json.load(f)
                weekly_schedule_data = news_data.get("weekly_schedule", [])
            except Exception:
                pass

    if weekly_schedule_data is None:
        weekly_schedule_data = []

    if lang == "en":
        est_schedule = build_english_weekly_schedule(weekly_schedule_data, date_str)
    else:
        est_schedule = build_korean_weekly_schedule(weekly_schedule_data, date_str)
    content = replace_weekly_schedule(content, est_schedule)

    # Collapse blank lines between a link and its body paragraph (only if the next non-empty line is a body paragraph)
    content = re.sub(
        r"(^|\n)(\[.*?\]\(https?://.*?\))\s*\n+(?=[^\[\s#\-*_])",
        r"\1\2\n",
        content,
    )

    lines = content.split("\n")
    formatted_lines = []
    in_weekly_schedule = False

    # 1. First pass: Handle headers, line-by-line formatting, earnings call conversion
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            formatted_lines.append(line)
            continue

        # Skip standalone metadata markers like "**Daily Point**"
        if re.search(r"^\s*\**Daily\s+Point\**\s*$", stripped, re.IGNORECASE):
            continue

        # Track if we are inside the Weekly Schedule section
        if stripped.startswith("### "):
            if "Weekly Schedule" in stripped or "주간 일정" in stripped:
                in_weekly_schedule = True
            else:
                in_weekly_schedule = False

        # 1.0) Bold removal for "Good day" or "안녕하세요"
        line = re.sub(
            r"\*\*(Good day|안녕하세요)(.*?)\*\*", r"\1\2", line, flags=re.IGNORECASE
        )
        stripped = line.strip()

        # 1.1) Header formatting: ## YYYYMMDD
        m_report = re.match(r"^##\s+(\d{4})(\d{2})(\d{2})$", stripped)
        if m_report:
            year, month, day = m_report.groups()
            if lang == "en":
                months_en = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                month_name = months_en[int(month) - 1]
                line = f"## {int(day)} {month_name} {year} Alpha Signal"
            else:
                line = f"## {year}년 {int(month)}월 {int(day)}일 Alpha Signal"

        # 1.2) Smart markdown line break ('<br />')
        # If this line is part of a list/schedule and the NEXT line is not empty and not separator
        date_pattern = r"^(\d+)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s*\("
        # date_pattern_ko: Matches Korean date syntax (e.g. "12월 25일 (목)") to inject paragraph breaks.
        date_pattern_ko = r"^(\d+)월\s+(\d+)일\s*\("

        is_target_line = (
            stripped.startswith(("_ ", "* ", "- ", "[", "★ ", "("))
            or re.match(date_pattern, stripped)
            or re.match(date_pattern_ko, stripped)
            or (in_weekly_schedule and not stripped.startswith("###"))
        )

        if is_target_line:
            # Check if next line is not empty and not a new section header or separator
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if (
                    next_stripped
                    and not next_stripped.startswith("###")
                    and not next_stripped == "---"
                ):
                    if not line.endswith("<br />"):
                        line += "<br />"

        formatted_lines.append(line)

    # 2. Second pass: Remove "---" below "Topline Signals" in "Daily Point"
    processed_lines = []
    in_daily_point = False
    found_topline = False
    for line in formatted_lines:
        stripped = line.strip()
        # Section detection
        if "### Daily Point" in line:
            in_daily_point = True
            found_topline = False
        elif in_daily_point and stripped.startswith("### "):
            in_daily_point = False
            found_topline = False

        if in_daily_point:
            # Check for Topline Signals
            if "**Topline Signals**" in line:
                found_topline = True

            # If we found topline and this line is exactly "---" or "***", skip it
            if found_topline and (stripped == "---" or "***" in stripped):
                continue
        processed_lines.append(line)

    # 3. Apply newline formatting constraints
    # Ensure Good day. / 안녕하세요. is followed by exactly two newlines (1 blank line)
    formatted_text = "\n".join(processed_lines)
    formatted_text = re.sub(
        r"((?:Good day\.|안녕하세요\.))\s*\n+",
        r"\1\n\n",
        formatted_text,
        flags=re.IGNORECASE,
    )

    # Collapse 3 or more consecutive newlines to 2 newlines (1 blank line)
    formatted_text = re.sub(r"\n{3,}", "\n\n", formatted_text)

    # 4. Escape raw dollars (except those already escaped) to prevent LaTeX rendering issues
    formatted_text = re.sub(r"(?<!\\)\$", r"\$", formatted_text)

    # 5. Format category slashes with spaces for better visual representation in final reports
    formatted_text = formatted_text.replace("AI/Robotics/EV", "AI / Robotics / EV")
    formatted_text = formatted_text.replace("Power/Grid", "Power / Grid")
    formatted_text = formatted_text.replace("Consumer/Retail", "Consumer / Retail")

    return formatted_text


def run_formatter(input_file: str, output_file: str, lang: str = "en"):
    """Reads input_file, formats it using format_content, and saves to output_file."""
    logger.info(f"Starting formatting process: {input_file} -> {output_file} ({lang})")
    if not os.path.exists(input_file):
        logger.error(f"Error: Input file {input_file} does not exist.")
        return False

    try:
        # Try loading weekly schedule JSON
        weekly_schedule_data = None
        date_part = None
        m = re.search(r"(\d{8})", input_file)
        if m:
            date_part = m.group(1)

        if date_part:
            data_dir = os.path.dirname(input_file)
            news_file = os.path.join(data_dir, f"daily_news_{date_part}.json")
            if not os.path.exists(news_file):
                import glob

                news_files = sorted(
                    glob.glob(os.path.join(data_dir, "daily_news_*.json"))
                )
                if news_files:
                    news_file = news_files[-1]

            if os.path.exists(news_file):
                try:
                    with open(news_file, "r", encoding="utf-8") as f:
                        news_data = json.load(f)
                    weekly_schedule_data = news_data.get("weekly_schedule", [])
                    logger.info(
                        f"Successfully loaded weekly schedule data ({len(weekly_schedule_data)} items) from {news_file}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not load weekly schedule JSON inside run_formatter: {e}"
                    )

        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        formatted = format_content(content, lang, weekly_schedule_data)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted)

        logger.info("Successfully formatted report!")
        return True
    except Exception as e:
        logger.error(f"Error during formatting: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buddy Core Report Formatter")
    parser.add_argument("--input", required=True, help="Path to input text report")
    parser.add_argument(
        "--output", required=True, help="Path to output markdown report"
    )
    parser.add_argument(
        "--lang", default="en", choices=["en", "ko"], help="Target language (en/ko)"
    )

    args = parser.parse_args()
    run_formatter(args.input, args.output, args.lang)
