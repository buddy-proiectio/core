import json
import os
import sys


def sort_articles_by_category(articles: list) -> dict:
    """
    Sorts a list of article dictionaries into 8 specific categories based on
    the 'matched_keywords' field. Uses case-insensitive, first-match routing.
    """
    # 1. The Routing Map
    routing_map = {
        "General": ["FOMC", "CPI", "Fed", "Interest rate"],
        "Bitcoin": ["Bitcoin", "BTC"],
        "Semiconductor": ["Nvidia", "NVDA", "Broadcom", "AVGO", "Micron", "MU", "AMD"],
        "AI": ["Anthropic", "OpenAI"],
        "Bio": ["Eli Lilly", "LLY", "Novo Nordisk", "NVO", "FDA"],
        "Space": ["SpaceX", "NASA"],
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

    # Pre-process the routing map into a flat, lowercase dictionary for O(1) lookups
    keyword_to_category = {}
    for category, keywords in routing_map.items():
        for kw in keywords:
            keyword_to_category[kw.lower()] = category

    # Initialize the output structure with all 8 category keys mapped to empty lists
    categorized_articles = {category: [] for category in routing_map.keys()}

    # 2. Routing Logic
    for article in articles:
        matched_keywords = article.get("matched_keywords", [])
        assigned = False

        # Iterate through the article's matched keywords
        for kw in matched_keywords:
            category = keyword_to_category.get(kw.lower())

            if category:
                # Collision Rule: First match wins
                categorized_articles[category].append(article)
                assigned = True
                break

        # Fallback: If no keywords matched (or list was empty), assign to "기타"
        if not assigned:
            categorized_articles["Others"].append(article)

    # 3. Output Structure
    return categorized_articles


if __name__ == "__main__":
    # Test script loading the provided daily news file
    # Adjust path to the root folder where the JSON file lives.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_json_path = os.path.join(base_dir, "daily_news_20260223.json")

    if not os.path.exists(target_json_path):
        print(f"Error: Could not find {target_json_path}")
        sys.exit(1)

    print(f"Loading {target_json_path}")

    with open(target_json_path, "r", encoding="utf-8") as f:
        daily_news = json.load(f)

    articles = daily_news.get("articles", [])
    print(f"Found {len(articles)} articles to sort.")

    sorted_data = sort_articles_by_category(articles)

    # Print the summary stats
    print("\n--- Sorting Summary ---")
    for category, items in sorted_data.items():
        print(f"{category}: {len(items)} articles")

    # Extract the YYYYMMDD date from the filename: "daily_news_20260223.json" -> "20260223"
    filename = os.path.basename(target_json_path)
    date_str = filename.replace("daily_news_", "").replace(".json", "")

    # Export the 8 distinct JSON files
    print("\n--- Exporting Files ---")
    for category, items in sorted_data.items():
        # Exception logic: Do not generate file if no articles exist
        if not items:
            continue

        # Clean the category name for safe file paths (though Korean characters are fine)
        safe_category = category.replace(" ", "_")
        out_filename = f"{safe_category}_sorted_{date_str}.json"
        out_path = os.path.join(base_dir, out_filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_filename}")
