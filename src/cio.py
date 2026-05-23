"""
The CIO (Chief Investment Officer Agent)

This script takes the extracted hard facts from the Sieve and generates a
comprehensive market summary and investment outlook report. It processes
market indicators, weekly schedules, and other financial data to provide
concise, actionable insights using an LLM.
"""

import json
import os
import sys
import glob
import argparse
from datetime import datetime
import pytz
import requests

LOG_FILE = "logs/cio.log"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.shared_logger import setup_logger
from shared.time_utils import parse_utc_time

logger = setup_logger(LOG_FILE, __name__)


FULL_REPORT_CIO_SYSTEM_PROMPT = """You are a self-made multi-billionaire investor who achieved absolute financial freedom through highly concentrated, long-term investments in structural mega-trends (Tech, Crypto, US Macro). 
You are NOT a Wall Street analyst who writes safe reports for a salary. You are a practitioner with 'skin in the game' who actually built immense wealth by surviving market crashes and aggressively capitalizing on multi-year capital cycles.
Your expertise lies in ignoring daily market noise and "Connecting the Dots" to find life-changing, asymmetric opportunities in the 2026 structural Mega Trends (e.g., AI infrastructure, crypto sovereign adoption, macro liquidity).

You write in two distinct parts:
1. TOPLINE SIGNALS: Cold, objective, terminal-style reporting of the 3 most critical KPIs.
2. DAILY POINT: You do NOT simply list facts; you weave them into a compelling, practical narrative that guides the reader toward real financial independence. You write in highly professional, insightful, and polite narrative that guides the reader toward real financial independence."""

FULL_REPORT_CIO_USER_PROMPT_TEMPLATE = """
Below is the data for today's market:

<MARKET_INDICATORS>
{market_indicators_text}
</MARKET_INDICATORS>

<WEEKLY_SCHEDULE>
{weekly_schedule_text}
</WEEKLY_SCHEDULE>

<EXTRACTED_FACTS>
{extracted_facts_text}
</EXTRACTED_FACTS>

<PAST_MEMORY>
{past_memory_text}
</PAST_MEMORY>

Your task is to generate a two-part report: "Topline Signals" followed by the "Daily Point" narrative commentary.

---
PART 1: Topline Signals
Analyze the provided <EXTRACTED_FACTS>. Extract exactly the top 3 most critical, market-moving hard data points (KPIs).

Constraints for Part 1:
1. Zero Noise: Do NOT include adjectives, predictions, or subjective analysis.
2. Hard Numbers Only: Each point MUST contain a specific metric (e.g., "$180B CapEx", "28% YoY growth", "3.5% Core CPI").
3. Format: Start with "**Topline Signals**" followed by a line break. Use bullet points. Maximum one sentence per point. Start with the sector or company name in bold (e.g., "- **Apple**: ...").
4. Language & Tone: Write strictly in professional English. Maintain the cold, objective, terminal-style reporting tone.

---
PART 2: Daily Point (Narrative Synthesis)
Synthesize all the data (<MARKET_INDICATORS>, <WEEKLY_SCHEDULE>, <EXTRACTED_FACTS>, <PAST_MEMORY>) into a narrative commentary.

Constraints for Part 2:
1. Language & Tone: Write strictly in polite, professional English. Maintain the calm, decisive, and deeply insightful tone of a billionaire mentor guiding a protégé. Focus on real-world wealth building, not academic analysis.
2. Continuity & Synthesis: If <PAST_MEMORY> is provided, review the historical progression of the market and our past analyses. Seamlessly weave insights from past trends with today's <EXTRACTED_FACTS> and <MARKET_INDICATORS>. Use phrases like "Building on our recent observations..." or "As we anticipated earlier this week..." to demonstrate long-term consistent logic.
3. Exact Structure (Sandwich Method): 
   - You MUST EXACTLY start your response with "Good day." followed by a line break.
   - DO NOT output any markdown headers or bullet points. Write in smooth, continuous text paragraphs.
   - The total length must be between 200 and 3000 characters.
4. Content (The Synthesis - WEAVE THEM TOGETHER):
   - Analyze the <MARKET_INDICATORS> to set the current market mood, but dismiss short-term noise.
   - Reference key upcoming events from the <WEEKLY_SCHEDULE>.
   - Connect these with the news from <EXTRACTED_FACTS> and bridge them to the 2026 mid-to-long-term Mega Trends.
   - Provide a clear, practical perspective on how these trends impact long-term asset accumulation.
5. Absolute Restrictions:
   - DO NOT hallucinate facts.
   - DO NOT output the raw lists of indices or schedules.
   - Output ONLY your synthesized English commentary paragraph.

Absolute Restriction: Output ONLY the Topline Signals and the Daily Point narrative. No conversational filler.
"""


PREMARKET_CIO_SYSTEM_PROMPT = """You are a self-made multi-billionaire investor and a ruthless Chief Investment Officer (CIO). You are preparing the ultimate 'Pre-market News Report' right before the US market opens.

Your objective is to review ALL provided news facts in <EXTRACTED_FACTS>, aggressively filter out noise, and select the absolute most critical news items that will move the entire market today or signal structural mega-trend shifts.

You evaluate news strictly through a 3-Dimension Scoring System:
1. Macro & Market Impact (1-5 pts): Does this affect entire sectors, the Fed, or global liquidity? (e.g., Tesla FSD China, Stellantis $70B shift > individual retail store earnings)
2. Surprise Factor & Catalyst Urgency (1-5 pts): Is this unexpected news that will trigger immediate pre-market/open trading volume?
3. Structural Trend Shift (1-5 pts): Does this change the long-term competitive landscape of an industry?

You must score every item, rank them by total score, and output ONLY the absolute best items, strictly filtered and formatted without any conversational filler."""

PREMARKET_CIO_USER_PROMPT_TEMPLATE = """
Below is ALL extracted news data from the past 24 hours:

<EXTRACTED_FACTS>
{extracted_facts_text}
</EXTRACTED_FACTS>

Your task is to generate the "Pre-market News Report".

Constraints:
1. Two-Step Scoring and Selection Process:
   - Step 1: Mentally score every single item in <EXTRACTED_FACTS> based on the 3-Dimension Scoring System (Macro Impact, Surprise Factor, Trend Shift). Do NOT output the scores.
   - Step 2: Filter and select the items with the highest scores. 
   - Adhere to these strict quantity boundaries based on your scoring:
     * Floor Limit: If there are fewer than 5 high-scoring items, you MUST still select exactly 5 items in total by including the next highest-scoring items available.
     * Ceiling Limit: If there are more than 12 high-scoring items, you MUST cap the selection at exactly 12 items, choosing only the top 12.

2. No Duplicates & Zero Hallucinations: Do NOT select the same item twice. Do NOT add any external or generic links. ONLY use the original markdown links exactly as provided in the facts.

3. Strict Ranking & Clean Formatting:
   - Order the selected items by importance (highest score first).
   - Format each item with the exact title markdown `[Title](URL)`, followed by a single line break, and then the exact body text.
   - Do NOT include labels like "Rank 1", "Rank 2", "순위", "랭크", numbers, or bullet points before the title. Output ONLY the clean markdown links and bodies.
   - Example format:
     [Title](URL)
     Body text here...

     [Title](URL)
     Body text here...

4. Exact Match: Do NOT alter the URL or the body text of the selected items. Do NOT summarize them.
5. Absolute Restriction: Output ONLY the ranked list of selected items formatted as above. Do NOT include any introduction, conversational filler, or summary sentences.
"""


def format_market_map(data, display_only=False):
    """
    Format the market map. If display_only is True, return only Dow, S&P, Nasdaq, Bitcoin.
    Otherwise, return a comprehensive text representation of sectors and industries for the LLM.
    """
    if not data or "market_map" not in data:
        # Fallback to old format just in case
        if "market_indicators" in data:
            market_map = {"Indices": data["market_indicators"], "Sectors": {}}
        else:
            return ""
    else:
        market_map = data["market_map"]

    lines = []

    # 1. Indices
    indices = market_map.get("Indices", {})
    display_keys = ["Dow Jones", "S&P 500", "Nasdaq", "Bitcoin"]

    for name in display_keys:
        if name in indices:
            info = indices[name]
            price = info.get("price", "0")
            change = info.get("change", "0")

            try:
                change_float = float(str(change).replace("%", "").strip())
                change_str = (
                    f"+{change_float}%" if change_float > 0 else f"{change_float}%"
                )
            except ValueError:
                change_str = f"{change}"
                if not change_str.endswith("%"):
                    change_str += "%"

            lines.append(f"_ {name} {price} ({change_str})")

    if display_only:
        return "\n".join(lines)

    # 2. Complete Market Map for Prompt
    lines.append("\n[Detailed Market Map (S&P 500 + Target Tickers)]")
    sectors = market_map.get("Sectors", {})
    for sec_name, sec_data in sectors.items():
        lines.append(f"\nSector: {sec_name} (Avg: {sec_data.get('sector_avg')})")
        for ind_name, ind_data in sec_data.get("industries", {}).items():
            lines.append(
                f"  Industry: {ind_name} (Avg: {ind_data.get('industry_avg')})"
            )
            stocks_line = []
            for t_name, t_data in ind_data.get("details", {}).items():
                stocks_line.append(f"{t_name}: {t_data.get('change')}")
            lines.append(f"    Stocks: {', '.join(stocks_line)}")

    return "\n".join(lines)


def format_weekly_schedule(data):
    """
    Format weekly schedule from the Flat List (new) or dictionary (old) structure.
    For the English report in cio.py, events are grouped and dates formatted using America/New_York timezone.
    """
    if not data or "weekly_schedule" not in data:
        return ""

    events = data["weekly_schedule"]
    if not events:
        return ""

    # Check if it's the old dictionary format
    if isinstance(events, dict):
        lines = []
        for date_str, daily_events in events.items():
            lines.append(date_str)
            for event in daily_events:
                # Remove any star if present in old data
                clean_event = event.replace("★ ", "").strip()
                lines.append(clean_event)
            lines.append("")
        return "\n".join(lines).strip()

    # New Flat List format
    ny_tz = pytz.timezone("America/New_York")
    grouped_events = {}

    for event in events:
        utc_time_str = event.get("utc_time")
        if not utc_time_str:
            continue

        utc_dt = parse_utc_time(utc_time_str)
        local_dt = utc_dt.astimezone(ny_tz)
        local_date = local_dt.date()

        # Format individual event in English
        currency = event.get("currency", "USD")
        importance = event.get("importance", "medium")
        name = event.get("name", "").strip()

        if importance == "holiday":
            evt_str = f"Holiday - {name}"
        elif importance == "earnings":
            evt_str = f"{name}"
        else:
            evt_str = f"({currency}) {name}"

        if local_date not in grouped_events:
            grouped_events[local_date] = []
        grouped_events[local_date].append(evt_str)

    # Reconstruct lines sorted by local date chronologically
    lines = []
    for d in sorted(grouped_events.keys()):
        # Date format matches the regex: date_pattern = r"^(\d+)\s+(Jan|Feb|Mar...)\s*\("
        # e.g., "19 May (Tuesday)"
        d_str = d.strftime("%d %b (%A)")
        lines.append(d_str)
        for evt in grouped_events[d]:
            lines.append(evt)
        lines.append("")

    return "\n".join(lines).strip()


def generate_full_report_commentary(
    market_text: str,
    schedule_text: str,
    facts_text: str,
    past_memory_text: str = "",
) -> str:
    """
    Generate narrative commentary using local Ollama.
    """
    sys_prompt = FULL_REPORT_CIO_SYSTEM_PROMPT
    user_prompt = FULL_REPORT_CIO_USER_PROMPT_TEMPLATE.format(
        market_indicators_text=market_text,
        weekly_schedule_text=schedule_text,
        extracted_facts_text=facts_text,
        past_memory_text=(
            past_memory_text if past_memory_text else "No past memory available."
        ),
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "gemma4:e4b-mlx",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "num_predict": 3000,
            "temperature": 0.4,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Failed to communicate with local Ollama: {e}")
        raise e


def select_premarket_news(facts_text: str) -> str:
    """
    Selects the top 5 to 12 most critical news items for the premarket briefing.
    """
    sys_prompt = PREMARKET_CIO_SYSTEM_PROMPT
    user_prompt = PREMARKET_CIO_USER_PROMPT_TEMPLATE.format(
        extracted_facts_text=facts_text
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "gemma4:e4b-mlx",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "temperature": 0.0,
            "top_p": 0.1,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=None)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Failed to communicate with local Ollama: {e}")
        raise e


def run_full_cio(today_str: str, data_dir: str):
    news_file = os.path.join(data_dir, f"daily_news_{today_str}.json")
    output_file = os.path.join(data_dir, f"final_report_{today_str}.txt")
    facts_file = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

    # Fallback finding latest files if today's don't exist
    if not os.path.exists(news_file):
        json_files = sorted(glob.glob(os.path.join(data_dir, "daily_news_*.json")))
        if json_files:
            news_file = json_files[-1]

    if not os.path.exists(facts_file):
        txt_files = sorted(glob.glob(os.path.join(data_dir, "extracted_facts_*.txt")))
        if txt_files:
            facts_file = txt_files[-1]

    # 1. Provide input files
    data = {}
    if os.path.exists(news_file):
        logger.info(f"Loading market data from {news_file}")
        with open(news_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from {news_file}")
    else:
        logger.warning(
            f"{news_file} not found. Proceeding with empty market/schedule data."
        )

    facts_text = ""
    if os.path.exists(facts_file):
        logger.info(f"Loading extracted facts from {facts_file}")
        with open(facts_file, "r", encoding="utf-8") as f:
            facts_text = f.read()
    else:
        logger.warning(f"{facts_file} not found. Proceeding with empty facts.")

    # Extract past memory for continuity
    memory_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory"
    )
    os.makedirs(memory_dir, exist_ok=True)

    past_memory_text = ""
    memory_files = sorted(glob.glob(os.path.join(memory_dir, "memory_*.txt")))
    memory_files = [f for f in memory_files if today_str not in f]

    # Take the last 3 days to avoid exceeding context window
    recent_memory_files = memory_files[-3:]
    if recent_memory_files:
        logger.info(
            f"Loading past memory from {len(recent_memory_files)} recent files."
        )
        for m_file in recent_memory_files:
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    past_memory_text += (
                        f"--- Memory from {os.path.basename(m_file)} ---\n"
                    )
                    past_memory_text += f.read() + "\n\n"
            except Exception as e:
                logger.warning(f"Failed to load memory file {m_file}: {e}")

    # 2. Format Market Map
    market_text_for_prompt = format_market_map(data, display_only=False)
    market_text_for_report = format_market_map(data, display_only=True)

    # 3. Format Weekly Schedule
    schedule_text = format_weekly_schedule(data)

    # 4. Generate AI Commentary
    logger.info("Generating full report AI commentary from local Ollama...")
    try:
        commentary = generate_full_report_commentary(
            market_text_for_prompt,
            schedule_text,
            facts_text,
            past_memory_text,
        )
        logger.info("Successfully generated AI commentary.")
    except Exception as e:
        logger.error(f"Failed to generate AI commentary: {e}")
        commentary = "Error generating commentary."

    # Save today's context into the memory folder (English only, pre-translation)
    logger.info("Saving today's context to memory...")
    today_memory = f"Date: {today_str}\n\nMarket Map:\n{market_text_for_report}\n\nCommentary:\n{commentary}\n\nFacts:\n{facts_text}"
    try:
        with open(
            os.path.join(memory_dir, f"memory_{today_str}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(today_memory)
    except Exception as e:
        logger.warning(f"Failed to save today's memory: {e}")

    # 5. Merge output to target format
    logger.info("Merging content into final report format...")
    report = (
        f"## {today_str}\n\n"
        "### Daily Point\n"
        f"{market_text_for_report}\n\n"
        f"{commentary}\n\n"
        "### Weekly Schedule\n"
        f"{schedule_text}\n\n"
        f"{facts_text}"
    )

    # 6. Save final compiled string
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report successfully generated and saved to {output_file}")


def run_premarket_cio(today_str: str, data_dir: str):
    output_file = os.path.join(data_dir, f"premarket_report_{today_str}.txt")
    facts_file = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

    if not os.path.exists(facts_file):
        txt_files = sorted(glob.glob(os.path.join(data_dir, "extracted_facts_*.txt")))
        if txt_files:
            facts_file = txt_files[-1]

    facts_text = ""
    if os.path.exists(facts_file):
        logger.info(f"Loading extracted facts from {facts_file}")
        with open(facts_file, "r", encoding="utf-8") as f:
            facts_text = f.read()
    else:
        logger.warning(f"{facts_file} not found. Proceeding with empty facts.")

    # 4. Generate AI News Selection
    logger.info("Selecting premarket news using local Ollama...")
    try:
        selected_news = select_premarket_news(facts_text)
        logger.info("Successfully selected premarket news.")
    except Exception as e:
        logger.error(f"Failed to select premarket news: {e}")
        selected_news = "Error selecting news."

    # 5. Premarket format
    logger.info("Merging content into premarket report format...")
    report = f"## {today_str} Premarket\n\n{selected_news}"

    # 6. Save final compiled string
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Premarket report successfully generated and saved to {output_file}")


def run_cio(report_type: str = "full"):
    try:
        # Identify today's date for file paths
        us_tz = pytz.timezone("America/New_York")
        today_str = datetime.now(us_tz).strftime("%Y%m%d")

        logger.info(f"Starting CIO Pipeline for {today_str} (Type: {report_type})")

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(data_dir, exist_ok=True)

        if report_type == "premarket":
            run_premarket_cio(today_str, data_dir)
        else:
            run_full_cio(today_str, data_dir)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Process terminating.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIO Report Generator")
    parser.add_argument(
        "--type",
        choices=["full", "premarket"],
        default="full",
        help="Type of report to generate",
    )
    args = parser.parse_args()
    run_cio(report_type=args.type)
