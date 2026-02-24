# Buddy Core Pipeline

Buddy Core is an automated, multi-agent Intelligence Pipeline designed to act as a personal multi-billionaire Chief Investment Officer (CIO). The system monitors global news via RSS, tracks market indicators, extracts deeply actionable facts using LLMs, and synthesizes them into a highly curated, professionally translated daily report.

## Project Architecture

The pipeline consists of five key modules, each handling a specific stage of data processing:

### 1. Sieve (`sieve.py`)

**Role:** Data Gathering & Rule-based Filtering Bot

- Continuously fetches data from configured RSS feeds, Yahoo Finance (for market indicators), and Finnhub (for earnings/macro events).
- Applies rigorous keyword/regex filtering and uses semantic deduplication to avoid redundant news.
- Dumps a master payload of the day's events into a timestamped JSON file (e.g., `daily_news_YYYYMMDD.json`) at a set time (e.g., 04:50 AM).

### 2. Sorter (`sorter.py`)

**Role:** Article Categorization

- Scans the raw daily news pool.
- Categorizes articles into predefined structural mega-trends (e.g., AI, Crypto, Tech, US Macro) based on exact keyword matches.
- Saves segmented JSON files (e.g., `AI_sorted_YYYYMMDD.json`) for specialized tracking or agents.

### 3. Extractor (`extractor.py`)

**Role:** AI Fact Extraction Engine

- Iterates over the filtered articles provided by the Sieve.
- Employs an LLM (via LiteLLM) to extract only the hard facts, KPIs, and critical structural shifts.
- Ensures no hallucinated data and strict formatting, saving the output to `extracted_facts_YYYYMMDD.txt`.

### 4. CIO (`cio.py`)

**Role:** Insight Synthesis & Report Generation

- Adopts the persona of a highly successful, self-made investor.
- Weaves together the extracted facts, market indicators, and weekly schedule into a cohesive, insightful morning commentary ("Daily Point").
- Formats everything into a structured report saved as `final_report_YYYYMMDD.txt`.

### 5. Translator (`translator.py`)

**Role:** Localization Engine

- Translates the final English report into professional Korean.
- Maintains strict formatting rules, specialized financial vocabulary, and specific Markdown line break (`\`) constraints.
- Overwrites or creates the finalized Korean markdown report ready for viewing or sending.

---

## Utilities & Infrastructure

- **`shared_logger.py`**: A centralized, color-coded logging utility used by all modules. It creates and routes module-specific log output directly into the `logs/` directory, preventing clutter in the root folder.
- **`logs/` Directory**: Contains the isolated logs for each module (`sieve.log`, `extractor.log`, `cio.log`, `sorter.log`, `translator.log`).

## Setup & Dependencies

1. Ensure you have the required Python packages installed (refer to `requirements.txt`).
2. Environment variables may be needed for specific APIs (e.g., `FINNHUB_API_KEY`)
3. Ollama must be running. Using model is `llama3.1`, `translategemma`.
4. You can execute each script sequentially once the morning's `Sieve` payload has been finalized.
