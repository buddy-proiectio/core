# Buddy Core Pipeline

Buddy Core is an automated, multi-agent Intelligence Pipeline designed to act as a personal multi-billionaire Chief Investment Officer (CIO). The system monitors global news via RSS, tracks market indicators, extracts deeply actionable facts using LLMs, and synthesizes them into highly curated, professionally translated daily reports.

## Project Architecture & Dual-Trigger System

The pipeline operates in a **Dual-Trigger System** (Full vs. Pre-market) with **Incremental processing** to optimize API costs and execution time.

### 1. Sieve (`sieve/`)

**Role:** Data Gathering & Rule-based Filtering Bot

- **Running Environment**: Deployed on an independent Oracle Cloud server, operating 24/7.
- Continuously fetches data from configured RSS feeds, Yahoo Finance, and Finnhub.
- Applies rigorous keyword/regex filtering and uses semantic deduplication to avoid redundant news.
- Dumps data into timestamped JSON files (`daily_news_YYYYMMDD.json` / `premarket_news_YYYYMMDD.json`).

### 2. Sorter (`src/sorter.py`)

**Role:** Article Categorization

- Categorizes articles into predefined structural mega-trends (e.g., AI, Crypto, Tech, US Macro) based on exact keyword matches.
- Saves segmented JSON files for specialized tracking or agents.

### 3. Extractor (`src/extractor.py`)

**Role:** AI Fact Extraction Engine & Incremental State Manager

- Employs an LLM (Local Ollama) to extract only hard facts, KPIs, and critical structural shifts.
- **Incremental State Management**: Persists processed URLs in `extracted_state_YYYYMMDD.json` to skip duplicate/failed extractions in subsequent runs, drastically saving LLM resources.
- Saves the output to `extracted_facts_YYYYMMDD.txt`.

### 4. CIO (`src/cio.py`)

**Role:** Insight Synthesis & Report Generation

- **Full Report (06:00)**: Weaves together extracted facts, market indicators, past memory, and weekly schedules into a cohesive, insightful morning commentary ("Daily Point"). Saves to `final_report_YYYYMMDD.txt`.
- **Premarket Report (08:30)**: Aggressively filters noise and selects exactly 5 to 12 most critical, unique news items. Uses strict LLM constraints without past memory overhead. Saves to `premarket_report_YYYYMMDD.txt`.

### 5. Translator (`src/translator.py`)

**Role:** Localization Engine

- Translates the final English reports into professional Korean utilizing Google Translate (`deep-translator` with `ko` target).
- Creates finalized Korean markdown reports:
  - Full Report: `alpha_signal_YYYYMMDD.md`
  - Premarket Report: `alpha_signal_premarket_YYYYMMDD.md`

### Orchestration (`src/__init__.py`)

**Role:** Automated Pipeline Execution & Data Lifecycle

- Operates in three modes: `incremental`, `full`, and `premarket`.
- Securely pulls Sieve's payload from Oracle Cloud via SCP.
- **Deferred Cleanup**: Intelligently delays intermediate data file deletion until the `premarket` run completes to ensure narrative continuity.
- **GitHub Automation**: Automatically commits and pushes the generated markdown reports to the GitHub repository.

---

## Utilities & Infrastructure

- **`src/prompts.py`**: Precise LLM prompt templates crafted to extract high-quality, actionable financial intelligence.
- **`shared/`**: Contains shared configurations such as `shared_logger.py` and `market_map_targets.json`.
- **`memory/`**: Houses long-term semantic context files (`memory_YYYYMMDD.txt`) for the CIO agent.
- **`logs/`**: Isolated logs for each module.
- **`scripts/launchd/`**: Contains macOS `.plist` scheduling files for local automation.

## Setup & Dependencies

1. **System Requirements**:
   - Remote server (Oracle Cloud) running `sieve/sieve.py` 24/7.
   - Local macOS environment running the pipeline via `launchd` tasks.

2. **Python Packages**:
   Install via `pip install -r requirements.txt`.
   Key dependencies: `feedparser`, `yfinance`, `sentence-transformers`, `deep-translator`.

3. **Environment & API Keys**:
   - Set up API keys (e.g., `FINNHUB_API_KEY`).
   - Configure Oracle Cloud SSH key path in `src/__init__.py`.

4. **Local LLM**:
   - Ollama engine must be active locally using the `llama3.1` model.

5. **Automation Configuration via macOS launchd**:
   - Copy the provided `.plist` files from `scripts/launchd/` to `~/Library/LaunchAgents/`.
   - Update the `StartCalendarInterval` blocks in the `.plist` files to match your exact local time for the NY schedule.
   - Load the schedules:
     ```bash
     launchctl load ~/Library/LaunchAgents/com.buddy.incremental.plist
     launchctl load ~/Library/LaunchAgents/com.buddy.full.plist
     launchctl load ~/Library/LaunchAgents/com.buddy.premarket.plist
     ```
   - _Note_: The `.plist` files utilize `/usr/bin/caffeinate -i` to prevent macOS from sleeping during execution.
