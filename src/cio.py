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
from datetime import datetime

# Eagerly import apscheduler and concurrent.futures to prevent lazy loading
# during script shutdown, which causes "can't register atexit after shutdown"
try:
    import concurrent.futures.process
    import apscheduler.schedulers.asyncio
except ImportError:
    pass

LOG_FILE = "logs/cio.log"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root to sys.path to allow importing from 'shared'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.shared_logger import setup_logger

logger = setup_logger(LOG_FILE, __name__)

import requests

CIO_SYSTEM_PROMPT = """You are a self-made multi-billionaire investor who achieved absolute financial freedom through highly concentrated, long-term investments in structural mega-trends (Tech, Crypto, US Macro). 
You are NOT a Wall Street analyst who writes safe reports for a salary. You are a practitioner with 'skin in the game' who actually built immense wealth by surviving market crashes and aggressively capitalizing on multi-year capital cycles.
Your expertise lies in ignoring daily market noise and "Connecting the Dots" to find life-changing, asymmetric opportunities in the 2026 structural Mega Trends (e.g., AI infrastructure, crypto sovereign adoption, macro liquidity).
You write in highly professional, insightful, and polite English.
You do NOT simply list facts; you weave them into a compelling, practical narrative that guides the reader toward real financial independence."""

CIO_USER_PROMPT_TEMPLATE = """
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

Your task is to write the "Daily Point" morning commentary synthesizing all the data above.

CRITICAL RULES:
1. Language & Tone: Write strictly in polite, professional English. Maintain the calm, decisive, and deeply insightful tone of a billionaire mentor guiding a protégé. Focus on real-world wealth building, not academic analysis.
2. Exact Structure (Sandwich Method): 
   - You MUST EXACTLY start your response with "Good morning." followed by a line break.
   - DO NOT output any markdown headers or bullet points. Write in smooth, continuous text paragraphs.
   - The total length must be between 200 and 3000 characters.
   - You MUST conclude your commentary with a thoughtful, profound piece of advice regarding life and marriage, reflecting the wisdom of a self-made billionaire who values not just wealth, but a successful personal life.
3. Content (The Synthesis - WEAVE THEM TOGETHER):
   - Analyze the <MARKET_INDICATORS> to set the current market mood, but dismiss short-term noise.
   - Reference key upcoming events from the <WEEKLY_SCHEDULE>.
   - Connect these with the news from <EXTRACTED_FACTS> and bridge them to the 2026 mid-to-long-term Mega Trends.
   - Provide a clear, practical perspective on how these trends impact long-term asset accumulation.
4. Absolute Restrictions:
   - DO NOT hallucinate facts.
   - DO NOT output the raw lists of indices or schedules.
   - Output ONLY your synthesized English commentary paragraph.
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
    Format weekly schedule directly from the dictionary without recalculating dates.
    """
    if not data or "weekly_schedule" not in data:
        return ""

    events = data["weekly_schedule"]
    lines = []

    for date_str, daily_events in events.items():
        lines.append(date_str)
        for event in daily_events:
            lines.append(event)
        # Add an empty line to separate days
        lines.append("")

    return "\n".join(lines).strip()


def generate_cio_commentary(
    market_text: str, schedule_text: str, facts_text: str
) -> str:
    """
    Generate narrative commentary using local Ollama.
    """
    user_prompt = CIO_USER_PROMPT_TEMPLATE.format(
        market_indicators_text=market_text,
        weekly_schedule_text=schedule_text,
        extracted_facts_text=facts_text,
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": CIO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": 3000,
            "num_ctx": 16384,
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


def run_cio():
    try:
        # Identify today's date for file paths
        today_str = datetime.now().strftime("%Y%m%d")

        logger.info(f"Starting CIO Pipeline for {today_str}")

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(data_dir, exist_ok=True)

        news_file = os.path.join(data_dir, f"daily_news_{today_str}.json")
        facts_file = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")
        output_file = os.path.join(data_dir, f"final_report_{today_str}.txt")

        # Fallback finding latest files if today's don't exist
        if not os.path.exists(news_file):
            json_files = sorted(glob.glob(os.path.join(data_dir, "daily_news_*.json")))
            if json_files:
                news_file = json_files[-1]

        if not os.path.exists(facts_file):
            txt_files = sorted(
                glob.glob(os.path.join(data_dir, "extracted_facts_*.txt"))
            )
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

        # 2. Format Market Map
        market_text_for_prompt = format_market_map(data, display_only=False)
        market_text_for_report = format_market_map(data, display_only=True)

        # 3. Format Weekly Schedule
        schedule_text = format_weekly_schedule(data)

        # 4. Generate AI Commentary
        logger.info("Generating AI commentary from local Ollama...")
        try:
            commentary = generate_cio_commentary(
                market_text_for_prompt, schedule_text, facts_text
            )
            logger.info("Successfully generated AI commentary.")
        except Exception as e:
            logger.error(f"Failed to generate AI commentary: {e}")
            commentary = "Error generating commentary."

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

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Process terminating.")
        return None
