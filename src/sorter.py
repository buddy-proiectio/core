"""
Sorter module for Buddy Core.

This module routes raw incoming news articles into distinct mega-trends (AI, Bitcoin,
Semiconductor, Aerospace, etc.) based on keyword scores defined in the configuration.
It dynamically binds to configurations and routing parameters from prompts.py.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # Negative keywords list to penalize non-investment/personal-finance noise
    negative_keywords = {
        "General": [
            "how to apply",
            "refinance your",
            "best credit card",
            "retirement calculator",
            "how to budget",
            "save money",
            "personal loan options",
            "mortgage tips for buyers",
            "lifestyle advice",
            "53y man",
            "porsche 911",
            "tips to retire",
            "personal finance",
            "how to retire",
            "credit card debt",
            "budgeting",
            "financial planner",
        ],
        "Software": ["ui update", "how to install", "tutorial", "best theme"],
    }

    # Load categories and keywords directly from prompts.py configuration
    routing_map = {
        category: config.get("keywords", {})
        for category, config in AGENT_CONFIGS.items()
    }

    # Pre-process the routing map into compiled regex patterns with weights
    category_patterns = {category: [] for category in routing_map.keys()}
    for category, keywords_data in routing_map.items():
        if isinstance(keywords_data, dict):
            for kw, weight in keywords_data.items():
                # Word boundary regex for exact term match (case insensitive)
                category_patterns[category].append(
                    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), weight)
                )
        else:
            # Fallback if list is provided
            for kw in keywords_data:
                category_patterns[category].append(
                    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), 1)
                )

    # Pre-process negative patterns into compiled regex patterns
    negative_patterns = {category: [] for category in negative_keywords.keys()}
    for category, keywords in negative_keywords.items():
        for kw in keywords:
            negative_patterns[category].append(
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            )

    # Initialize the output structure
    categorized_articles = {category: [] for category in routing_map.keys()}

    # 2. Routing Logic
    for article in articles:
        text_to_search = f"{article.get('title', '')} | {article.get('content', '')}"

        # Calculate a score for each category based on keyword frequencies and weights
        category_scores = {category: 0 for category in routing_map.keys()}

        for category, patterns in category_patterns.items():
            # Positive match additions
            for pattern, weight in patterns:
                matches = len(pattern.findall(text_to_search))
                category_scores[category] += matches * weight

            # Negative match subtractions (heavily penalize matching noise)
            if category in negative_patterns:
                for pattern in negative_patterns[category]:
                    neg_matches = len(pattern.findall(text_to_search))
                    category_scores[category] -= neg_matches * 5

        # Find the category with the highest score
        max_score = max(category_scores.values())

        if max_score > 0:
            # Get all categories that achieved the max score
            candidates = [
                cat for cat, score in category_scores.items() if score == max_score
            ]
            # Prioritize specific categories over 'Others' if there is a tie
            if len(candidates) > 1 and "Others" in candidates:
                candidates.remove("Others")
            best_category = candidates[0]
            categorized_articles[best_category].append(article)
        else:
            categorized_articles["Others"].append(article)

    # 3. Output Structure
    return categorized_articles


def run_sorter(report_type: str = "full", target_date: str | None = None):
    try:
        # Test script loading the provided daily news file
        # Adjust path to the root folder where the JSON file lives.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        target_json_path = None
        # Check if a json file is explicitly passed
        for arg in sys.argv[1:]:
            if arg.endswith(".json"):
                target_json_path = arg
                break

        if not target_json_path and target_date:
            if report_type == "premarket":
                target_json_path = os.path.join(
                    data_dir, f"premarket_news_{target_date}.json"
                )
            else:
                target_json_path = os.path.join(
                    data_dir, f"daily_news_{target_date}.json"
                )

        if not target_json_path:
            if report_type == "premarket":
                file_pattern = "premarket_news_*.json"
            else:
                file_pattern = "daily_news_*.json"

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
            safe_category = category.replace(" ", "_").replace("/", "_")
            out_filename = f"{safe_category}_sorted_{date_str}.json"
            out_path = os.path.join(data_dir, out_filename)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved: {out_filename}")

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Process terminating.")
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sorter Agent")
    parser.add_argument(
        "--type",
        choices=["full", "premarket"],
        default="full",
        help="Type of report to sort",
    )
    args = parser.parse_args()
    run_sorter(report_type=args.type)
