"""
The Chief Investment Officer (CIO) Agent

This script processes market indicators, weekly schedules, and extracted facts
to generate a highly professional, billionaire-mentor styled daily commentary (Daily Point)
and select the absolute most critical premarket news articles.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta
import pytz
import requests
from typing import Optional

LOG_FILE = "logs/cio.log"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.shared_logger import setup_logger
from shared.time_utils import parse_utc_time

logger = setup_logger(LOG_FILE, __name__)

# ==============================================================================
# 1. AI Prompts & Templates
# ==============================================================================

# Daily Point Narrative Commentary System Prompt
DAILY_COMMENTARY_SYSTEM_PROMPT = """You are a self-made multi-billionaire investor who achieved absolute financial freedom through highly concentrated, long-term investments in structural mega-trends (Tech, Crypto, US Macro). 
You are NOT a Wall Street analyst who writes safe reports for a salary. You are a practitioner with 'skin in the game' who actually built immense wealth by surviving market crashes and aggressively capitalizing on multi-year capital cycles.
Your expertise lies in ignoring daily market noise and "Connecting the Dots" to find life-changing, asymmetric opportunities in the 2026 structural Mega Trends (e.g., AI infrastructure, crypto sovereign adoption, macro liquidity).

You write in two distinct parts:
1. TOPLINE SIGNALS: Cold, objective, terminal-style reporting of the 3 most critical KPIs.
2. DAILY POINT: You do NOT simply list facts; you weave them into a compelling, practical narrative that guides the reader toward real financial independence. You write in highly professional, insightful, and polite narrative that guides the reader toward real financial independence."""

# Daily Point Narrative Commentary User Prompt Template
DAILY_COMMENTARY_USER_PROMPT_TEMPLATE = """
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
Synthesize all the data (<MARKET_INDICATORS>, <WEEKLY_SCHEDULE>, <EXTRACTED_FACTS>) into a narrative commentary.

Constraints for Part 2:
1. Language & Tone: Write strictly in polite, professional English. Maintain the calm, decisive, and deeply insightful tone of a billionaire mentor guiding a protégé. Focus on real-world wealth building, not academic analysis.
2. Continuity & Synthesis: Seamlessly weave insights from today's <EXTRACTED_FACTS> and <MARKET_INDICATORS>. Demonstrate professional, long-term consistent logic.
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

# Premarket News Selection System Prompt
PREMARKET_SELECTION_SYSTEM_PROMPT = """You are a self-made multi-billionaire investor and a ruthless Chief Investment Officer (CIO). You are preparing the ultimate 'Pre-market News Report' right before the US market opens.

Your objective is to review ALL provided news facts in <EXTRACTED_FACTS>, aggressively filter out noise, and select the absolute most critical news items that will move the entire market today or signal structural mega-trend shifts.

You evaluate news strictly through a 3-Dimension Scoring System:
1. Macro & Market Impact (1-5 pts): Does this affect entire sectors, the Fed, or global liquidity? (e.g., Tesla FSD China, Stellantis $70B shift > individual retail store earnings)
2. Surprise Factor & Catalyst Urgency (1-5 pts): Is this unexpected news that will trigger immediate pre-market/open trading volume?
3. Structural Trend Shift (1-5 pts): Does this change the long-term competitive landscape of an industry?

You must score every item, rank them by total score, and output ONLY the absolute best items, strictly filtered and formatted without any conversational filler."""

# Premarket News Selection User Prompt (Gemini ID-Selection Mode)
PREMARKET_SELECTION_USER_PROMPT_GEMINI = """
Below is ALL extracted news data from the past 24 hours. Each news item is assigned a unique numerical "ARTICLE ID" (e.g. 1, 2, 3...):

<EXTRACTED_FACTS_WITH_IDS>
{articles_text_for_prompt}
</EXTRACTED_FACTS_WITH_IDS>

Your task is to select the most critical news items for the "Pre-market News Report".

Constraints:
1. Two-Step Scoring and Selection Process:
   - Step 1: Evaluate every single item in <EXTRACTED_FACTS_WITH_IDS> based on the 3-Dimension Scoring System (Macro Impact, Surprise Factor, Trend Shift) out of a maximum of 15 points. Do NOT print or output the scores.
   - Step 2: Target and select items that score **12 points or higher** as your priority. Then, adhere to these strict quantity boundaries:
     * Floor Limit: If fewer than 5 items score 12+ points, you MUST still select exactly 5 items in total by padding the list with the next highest-scoring items available.
     * Ceiling Limit: If more than 12 items score 12+ points, you MUST cap the selection at exactly 12 items, choosing only the absolute top 12 highest-scoring items.

2. Strict Output Format (JSON only):
   - You MUST output your final selection strictly as a JSON object containing a single key "selected_ids", mapping to an array of numerical IDs in ranked order (highest score first).
   - Do NOT include any markdown formatting, triple backticks (e.g., ```json), conversational filler, comments, or extra keys. Just return the raw JSON object.
   - Example response format:
   {{"selected_ids": [3, 7, 12, 1, 5]}}
"""

# Premarket News Selection User Prompt (Local Ollama Text-to-Text Fallback Mode)
PREMARKET_SELECTION_USER_PROMPT_OLLAMA = """
Below is ALL extracted news data from the past 24 hours:

<EXTRACTED_FACTS>
{extracted_facts_text}
</EXTRACTED_FACTS>

Your task is to generate the "Pre-market News Report".

Constraints:
1. Two-Step Scoring and Selection Process:
   - Step 1: Evaluate every single item in <EXTRACTED_FACTS> based on the 3-Dimension Scoring System (Macro Impact, Surprise Factor, Trend Shift) out of a maximum of 15 points. Do NOT print or output the scores.
   - Step 2: Target and select items that score **12 points or higher** as your priority. Then, adhere to these strict quantity boundaries:
     * Floor Limit: If fewer than 5 items score 12+ points, you MUST still select exactly 5 items in total by padding the list with the next highest-scoring items available.
     * Ceiling Limit: If more than 12 items score 12+ points, you MUST cap the selection at exactly 12 items, choosing only the absolute top 12 highest-scoring items.

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

# ==============================================================================
# 2. Helper Functions
# ==============================================================================


def format_market_map(data: dict, display_only: bool = False) -> str:
    """
    Format the market map. If display_only is True, return only Dow, S&P, Nasdaq, Bitcoin.
    Otherwise, return a comprehensive text representation of sectors and industries for the LLM.
    """
    if not data or "market_map" not in data:
        if "market_indicators" in data:
            market_map = {"Indices": data["market_indicators"], "Sectors": {}}
        else:
            return ""
    else:
        market_map = data["market_map"]

    lines = []

    # 1. Major Indices
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

    # 2. Complete Sector/Industry Heatmap for AI Context
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


def format_weekly_schedule(data: dict, today_str: Optional[str] = None) -> str:
    """
    Format weekly schedule into a strict 7-day rolling calendar starting from today_str.
    For the English report, events are grouped and dates formatted using America/New_York timezone.
    All-day UTC midnight (00:00:00) events keep their date untouched to prevent timezone shifts.
    """
    events = []
    if data and "weekly_schedule" in data:
        events = data["weekly_schedule"] or []

    ny_tz = pytz.timezone("America/New_York")

    # Determine base date
    if today_str:
        try:
            base_date = datetime.strptime(today_str, "%Y%m%d").date()
        except ValueError:
            base_date = datetime.now(ny_tz).date()
    else:
        base_date = datetime.now(ny_tz).date()

    # Generate exactly 7 consecutive dates
    target_dates = [base_date + timedelta(days=i) for i in range(7)]
    grouped_events = {d: [] for d in target_dates}

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

        # Format event description
        currency = event.get("currency", "USD")
        importance = event.get("importance", "medium")
        name = event.get("name", "").strip()

        if importance == "holiday":
            evt_str = f"Holiday - {name}"
        elif importance == "earnings":
            evt_str = f"{name}"
        else:
            evt_str = f"({currency}) {name}"

        # Only group if it falls within the 7-day target range
        if local_date in grouped_events:
            grouped_events[local_date].append(evt_str)

    # Reconstruct lines sorted by local date chronologically
    lines = []
    for d in target_dates:
        d_str = d.strftime("%d %b (%A)")
        lines.append(d_str)
        for evt in grouped_events[d]:
            lines.append(evt)
        lines.append("")

    return "\n".join(lines).strip()


def call_gemini_api(
    sys_prompt: str, user_prompt: str, response_mime_type: str = "text/plain"
) -> str:
    """
    Calls the Gemini API (REST endpoint) using the provided system and user prompts.
    Tries early access gemini-3.5-flash first, then falls back gracefully to previous generations.
    """
    api_key = os.environ.get(
        "GEMINI_API_KEY", "AIzaSyBUVV6-n_nbHW84hg8blEegy8jtEYBPf4g"
    )

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]

    last_err = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": sys_prompt}]},
            "generationConfig": {
                "temperature": 0.0 if response_mime_type == "application/json" else 0.4,
                "responseMimeType": response_mime_type,
            },
        }

        try:
            logger.info(f"Calling Gemini API with model: {model} ...")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code == 404:
                logger.warning(f"Model {model} returned 404. Trying fallback...")
                continue
            response.raise_for_status()
            res_data = response.json()

            candidates = res_data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned from Gemini API")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No parts found in the first candidate")

            content_text = parts[0].get("text", "").strip()
            return content_text
        except Exception as e:
            logger.error(f"Error calling Gemini API with model {model}: {e}")
            last_err = e
            continue

    if last_err:
        raise last_err
    else:
        raise ValueError("Failed to call Gemini API: All models failed.")


def clean_json_response(response_text: str) -> str:
    """
    Cleans any markdown blocks (```json ... ```) from the LLM response.
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def parse_facts_into_articles(facts_text: str) -> list[dict]:
    """
    Parses facts_text (extracted_facts_YYYYMMDD.txt) into a list of articles.
    """
    blocks = re.split(r"\n\n+", facts_text)
    articles = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("### "):
            continue

        match = re.match(r"^\[(.*?)\]\((.*?)\)(?:\n([\s\S]*))?$", block)
        if match:
            title = match.group(1).strip()
            url = match.group(2).strip()
            body = match.group(3).strip() if match.group(3) else ""
            articles.append(
                {"title": title, "url": url, "body": body, "raw_block": block}
            )

    return articles


def generate_daily_commentary_gemini(
    market_text: str,
    schedule_text: str,
    facts_text: str,
) -> str:
    """
    Generate Daily Point narrative commentary using Gemini 3.5 Flash.
    """
    sys_prompt = DAILY_COMMENTARY_SYSTEM_PROMPT
    user_prompt = DAILY_COMMENTARY_USER_PROMPT_TEMPLATE.format(
        market_indicators_text=market_text,
        weekly_schedule_text=schedule_text,
        extracted_facts_text=facts_text,
    )

    return call_gemini_api(sys_prompt, user_prompt, response_mime_type="text/plain")


def generate_daily_commentary_ollama(
    market_text: str,
    schedule_text: str,
    facts_text: str,
) -> str:
    """
    Generate Daily Point narrative commentary using local Ollama.
    """
    sys_prompt = DAILY_COMMENTARY_SYSTEM_PROMPT
    user_prompt = DAILY_COMMENTARY_USER_PROMPT_TEMPLATE.format(
        market_indicators_text=market_text,
        weekly_schedule_text=schedule_text,
        extracted_facts_text=facts_text,
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
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


def select_premarket_news_gemini(facts_text: str) -> str:
    """
    Selects the top 5 to 12 most critical news items for the premarket briefing using Gemini 3.5 Flash.
    Parses facts into articles, assigns IDs, calls Gemini API in JSON Mode, and reconstructs the output in Python.
    """
    articles = parse_facts_into_articles(facts_text)
    if not articles:
        logger.warning("No parsed articles found in facts text.")
        return "No articles available for selection."

    formatted_articles = []
    for idx, art in enumerate(articles, 1):
        formatted_articles.append(
            f"--- ARTICLE ID: {idx} ---\n"
            f"Title: {art['title']}\n"
            f"URL: {art['url']}\n"
            f"Body: {art['body']}\n"
        )
    articles_text_for_prompt = "\n".join(formatted_articles)

    user_prompt = PREMARKET_SELECTION_USER_PROMPT_GEMINI.format(
        articles_text_for_prompt=articles_text_for_prompt
    )

    gemini_response = call_gemini_api(
        sys_prompt=PREMARKET_SELECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_mime_type="application/json",
    )

    cleaned_res = clean_json_response(gemini_response)
    logger.info(f"Received premarket response from Gemini: {cleaned_res}")
    res_data = json.loads(cleaned_res)
    selected_ids = res_data.get("selected_ids", [])

    selected_blocks = []
    for s_id in selected_ids:
        try:
            s_idx = int(s_id) - 1
            if 0 <= s_idx < len(articles):
                selected_blocks.append(articles[s_idx]["raw_block"])
        except (ValueError, TypeError):
            continue

    if not selected_blocks:
        raise ValueError("No valid article IDs were returned or resolved from Gemini.")

    selected_news = "\n\n".join(selected_blocks)
    logger.info(
        f"Successfully selected and merged {len(selected_blocks)} premarket news items using Gemini."
    )
    return selected_news


def select_premarket_news_ollama(facts_text: str) -> str:
    """
    Selects the top 5 to 12 most critical news items for the premarket briefing using Ollama.
    """
    sys_prompt = PREMARKET_SELECTION_SYSTEM_PROMPT
    user_prompt = PREMARKET_SELECTION_USER_PROMPT_OLLAMA.format(
        extracted_facts_text=facts_text
    )

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "num_predict": 2000,
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


# ==============================================================================
# 3. Main Executable Pipelines
# ==============================================================================


def run_full_cio(today_str: str, data_dir: str):
    news_file = os.path.join(data_dir, f"daily_news_{today_str}.json")
    output_file = os.path.join(data_dir, f"final_report_{today_str}.txt")
    facts_file = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

    # Fallback to finding the latest files if today's don't exist
    if not os.path.exists(news_file):
        json_files = sorted(glob.glob(os.path.join(data_dir, "daily_news_*.json")))
        if json_files:
            news_file = json_files[-1]

    if not os.path.exists(facts_file):
        txt_files = sorted(glob.glob(os.path.join(data_dir, "extracted_facts_*.txt")))
        if txt_files:
            facts_file = txt_files[-1]

    # Load market indicators and weekly calendar feed
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

    # Load semantic extracted facts
    facts_text = ""
    if os.path.exists(facts_file):
        logger.info(f"Loading extracted facts from {facts_file}")
        with open(facts_file, "r", encoding="utf-8") as f:
            facts_text = f.read()
    else:
        logger.warning(f"{facts_file} not found. Proceeding with empty facts.")

    # Format components
    market_text_for_prompt = format_market_map(data, display_only=False)
    market_text_for_report = format_market_map(data, display_only=True)
    schedule_text = format_weekly_schedule(data, today_str)

    # Generate Narrative Commentary
    logger.info("Generating full report AI commentary from Gemini 3.5 Flash...")
    try:
        commentary = generate_daily_commentary_gemini(
            market_text_for_prompt,
            schedule_text,
            facts_text,
        )
        logger.info("Successfully generated AI commentary using Gemini.")
    except Exception as e:
        logger.warning(
            f"Failed to generate AI commentary using Gemini API: {e}. Falling back to local Ollama..."
        )
        try:
            commentary = generate_daily_commentary_ollama(
                market_text_for_prompt,
                schedule_text,
                facts_text,
            )
            logger.info(
                "Successfully generated AI commentary using local Ollama fallback."
            )
        except Exception as fallback_err:
            logger.error(f"Full report commentary fallback also failed: {fallback_err}")
            commentary = "Error generating commentary."

    # Merge content into final report format
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

    # Select critical Premarket news items
    logger.info("Selecting premarket news using Gemini...")
    try:
        selected_news = select_premarket_news_gemini(facts_text)
    except Exception as e:
        logger.warning(
            f"Failed to select premarket news using Gemini API: {e}. Falling back to local Ollama..."
        )
        try:
            selected_news = select_premarket_news_ollama(facts_text)
            logger.info(
                "Successfully selected premarket news using local Ollama fallback."
            )
        except Exception as fallback_err:
            logger.error(f"Premarket selection fallback also failed: {fallback_err}")
            selected_news = "Error selecting news."

    # Compile premarket report
    logger.info("Merging content into premarket report format...")
    report = f"## {today_str} Premarket\n\n{selected_news}"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Premarket report successfully generated and saved to {output_file}")


def run_cio(report_type: str = "full"):
    try:
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
