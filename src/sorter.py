"""
The Sorter (Article Categorization & Routing Bot)

This script processes article data and organizes them into specific categories
based on matched keywords. It applies a routing map to sort articles into
predefined groups such as Bitcoin, AI, Semiconductor, and Software.
"""

import json
import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.shared_logger import setup_logger

LOG_FILE = "logs/sorter.log"

logger = setup_logger(LOG_FILE, __name__)


def sort_articles_by_category(articles: list) -> dict:
    """
    Sorts a list of article dictionaries into 8 specific categories based on
    the expanded keyword map from prompts.py, falling back to legacy keywords.
    Performs deep text search on title & content for more precise classification.
    """
    from prompts import AGENT_CONFIGS
    import re

    # 1. The Legacy Routing Map (Fallbacks for tickers)
    legacy_routing_map = {
        "General": ["FOMC", "CPI", "Fed", "Interest rate"],
        "Bitcoin": ["Bitcoin", "BTC", "Gold", "Silver"],
        "Semiconductor": ["Nvidia", "NVDA", "Broadcom", "AVGO", "Micron", "MU", "AMD"],
        "AI": ["Anthropic", "OpenAI"],
        "Bio": ["Eli Lilly", "LLY", "Novo Nordisk", "NVO", "FDA"],
        "Aerospace": ["SpaceX", "NASA"],
        "Software": [
            "Palantir",
            "PLTR",
            "Microsoft",
            "MSFT",
            "Meta",
            "META",
            "Oracle",
            "ORCL",
            "Netflix",
            "NFLX",
        ],
        "Others": [
            "Tesla",
            "TSLA",
            "Apple",
            "AAPL",
            "Amazon",
            "AMZN",
            "Walmart",
            "WMT",
        ],
    }

    # Merge prompts.py keywords and legacy keywords
    routing_map = {
        category: list(config.get("keywords", []))
        for category, config in AGENT_CONFIGS.items()
    }
    for category, keywords in legacy_routing_map.items():
        if category not in routing_map:
            routing_map[category] = []
        for kw in keywords:
            if kw.lower() not in [k.lower() for k in routing_map[category]]:
                routing_map[category].append(kw)

    # Pre-process the routing map into compiled regex patterns
    category_patterns = {category: [] for category in routing_map.keys()}
    for category, keywords in routing_map.items():
        for kw in keywords:
            # Word boundary regex for exact term match (case insensitive)
            category_patterns[category].append(
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            )

    # Initialize the output structure
    categorized_articles = {category: [] for category in routing_map.keys()}

    # 2. Routing Logic
    for article in articles:
        text_to_search = f"{article.get('title', '')} | {article.get('content', '')}"

        # Calculate a score for each category based on keyword frequencies
        category_scores = {category: 0 for category in routing_map.keys()}

        for category, patterns in category_patterns.items():
            for pattern in patterns:
                # Count all non-overlapping occurrences of the pattern
                matches = len(pattern.findall(text_to_search))
                category_scores[category] += matches

        # Find the category with the highest score
        best_category = max(category_scores, key=lambda k: category_scores[k])

        # If there's at least one match, assign to the best scoring category. Otherwise, fallback to 'Others'
        if category_scores[best_category] > 0:
            categorized_articles[best_category].append(article)
        else:
            categorized_articles["Others"].append(article)

    # 3. Output Structure
    return categorized_articles


def run_sorter(report_type: str = "full"):
    try:
        # Test script loading the provided daily news file
        # Adjust path to the root folder where the JSON file lives.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        if report_type == "premarket":
            file_pattern = "premarket_news_*.json"
        else:
            file_pattern = "daily_news_*.json"

        target_json_path = None
        # Check if a json file is explicitly passed
        for arg in sys.argv[1:]:
            if arg.endswith(".json"):
                target_json_path = arg
                break

        if not target_json_path:
            files = glob.glob(os.path.join(data_dir, file_pattern))
            if not files:
                logger.error(f"Error: Could not find any {file_pattern}")
                sys.exit(1)
            target_json_path = max(files, key=os.path.getmtime)
            logger.info(f"Auto-selected latest file: {target_json_path}")

        if not os.path.exists(target_json_path):
            logger.error(f"Error: Could not find {target_json_path}")
            sys.exit(1)

        logger.info(f"Loading {target_json_path}")

        with open(target_json_path, "r", encoding="utf-8") as f:
            daily_news = json.load(f)

        articles = daily_news.get("articles", [])
        logger.info(f"Found {len(articles)} articles to sort.")

        sorted_data = sort_articles_by_category(articles)

        # Print the summary stats
        logger.info("--- Sorting Summary ---")
        for category, items in sorted_data.items():
            logger.info(f"{category}: {len(items)} articles")

        # Extract the YYYYMMDD date from the filename: "daily_news_20260223.json" -> "20260223"
        filename = os.path.basename(target_json_path)
        date_str = (
            filename.replace("daily_news_", "")
            .replace("premarket_news_", "")
            .replace(".json", "")
        )

        # Export the 8 distinct JSON files
        logger.info("--- Exporting Files ---")
        for category, items in sorted_data.items():
            # Exception logic: Do not generate file if no articles exist
            if not items:
                continue

            # Clean the category name for safe file paths (spaces replaced with underscores)
            safe_category = category.replace(" ", "_")
            out_filename = f"{safe_category}_sorted_{date_str}.json"
            out_path = os.path.join(data_dir, out_filename)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved: {out_filename}")

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Process terminating.")
        sys.exit(0)
