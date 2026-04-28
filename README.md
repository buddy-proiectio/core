# Buddy Core Pipeline

Buddy Core is an automated, multi-agent Intelligence Pipeline designed to act as a personal multi-billionaire Chief Investment Officer (CIO). The system monitors global news via RSS, tracks market indicators, extracts deeply actionable facts using LLMs, and synthesizes them into a highly curated, professionally translated daily report.

## Project Architecture

The pipeline consists of five key modules and an orchestrator, each handling a specific stage of data processing:

### 1. Sieve (`sieve/`)

**Role:** Data Gathering & Rule-based Filtering Bot

- **Running Environment**: Deployed on an independent Oracle Cloud server, operating 24/7.
- Continuously fetches data from configured RSS feeds, Yahoo Finance (for market indicators), and Finnhub (for earnings/macro events).
- Applies rigorous keyword/regex filtering and uses semantic deduplication to avoid redundant news.
- Dumps a master payload of the day's events into a timestamped JSON file (e.g., `daily_news_YYYYMMDD.json`) at a set time (03:30 AM).

### 2. Sorter (`src/sorter.py`)

**Role:** Article Categorization

- Scans the raw daily news pool.
- Categorizes articles into predefined structural mega-trends (e.g., AI, Crypto, Tech, US Macro) based on exact keyword matches.
- Saves segmented JSON files (e.g., `AI_sorted_YYYYMMDD.json`) for specialized tracking or agents.

### 3. Extractor (`src/extractor.py`)

**Role:** AI Fact Extraction Engine

- Iterates over the filtered articles provided by the Sorter.
- Employs an LLM (Local Ollama) to extract only the hard facts, KPIs, and critical structural shifts.
- Ensures no hallucinated data and strict formatting, saving the output to `extracted_facts_YYYYMMDD.txt`.

### 4. CIO (`src/cio.py`)

**Role:** Insight Synthesis & Report Generation

- Adopts the persona of a highly successful, self-made investor.
- Weaves together the extracted facts, market indicators, and weekly schedule into a cohesive, insightful morning commentary ("Daily Point").
- Formats everything into a structured report saved as `final_report_YYYYMMDD.txt`.

### 5. Translator (`src/translator.py`)

**Role:** Localization Engine

- Translates the final English report into professional Korean utilizing Google Translate (`deep-translator`).
- Maintains strict formatting rules, specialized financial vocabulary, and specific Markdown line break (`\`) constraints.
- Overwrites or creates the finalized Korean markdown report (e.g., `alpha_signal_YYYYMMDD.md`).

### Orchestration (`src/__init__.py`)

**Role:** Automated Pipeline Execution

- Validates if the current date is a US trading day (skipping weekends and NYSE holidays).
- Securely pulls the Sieve's daily payload from the Oracle Cloud server via SCP.
- Sequentially processes the pipeline: **Sorter &rarr; Extractor &rarr; CIO &rarr; Translator**.
- Once the translated report is successfully generated, it cleans up all intermediate data files.

---

## Utilities & Infrastructure

- **`src/__init__.py`**: Pipeline orchestration script. Manages execution flow, checks for US trading days/holidays, pulls data from the cloud using SCP, runs the components in sequence, and handles final cleanups.
- **`src/prompts.py`**: Includes precise LLM prompt templates and rigorous configuration rules (e.g., `AGENT_CONFIGS`) specifically crafted to extract high-quality, actionable financial intelligence.
- **`shared/` Directory**: Contains shared configurations and utilities such as `shared_logger.py` (a centralized logging utility) and `market_map_targets.json`.
- **`memory/` Directory**: Houses the long-term semantic context files (e.g., `memory_YYYYMMDD.txt`) for the CIO agent to ensure longitudinal market analysis consistency.
- **`logs/` Directory**: Contains the isolated logs for each module (e.g., `sieve.log`, `extractor.log`).
- **`data/` Directory**: Staging ground for pipeline data. Holds intermediate JSONs, text files, and ultimately the finalized outputs.

## Setup & Dependencies

1.  **System Requirements**:
    - A remote server (e.g., Oracle Cloud) to consistently run `sieve/sieve.py` 24/7.
    - A distinct local environment (macOS) to run the morning processing via a `launchd` task scheduled strictly at 07:35 AM.

2.  **Python Packages**:
    Ensure the required Python packages are installed (`pip install -r requirements.txt`). Key dependencies include:
    - _Data Gathering (Sieve)_: `feedparser`, `requests`, `schedule`, `yfinance`, `trafilatura`
    - _Utilities_: `pytz`, `holidays`, `python-dateutil`
    - _AI Agents Component_: `sentence-transformers` for embeddings
    - _Translation Component_: `deep-translator`

3.  **Environment & API Keys**:
    - Specific services require valid environment variables or API keys (e.g., `FINNHUB_API_KEY`).
    - Setup an SSH key path inside `src/__init__.py` (e.g., `ORACLE_SSH_KEY`) to properly pull the daily payload from the Oracle Cloud.

4.  **Local LLM**:
    - The Ollama engine must be active and accessible locally. Model used in the pipeline includes `llama3.1`.

5.  **Automation Configuration via macOS launchd**:
    - Ensure the Sieve bot runs unattended on Oracle Server.
    - Configure a `launchd` agent locally (e.g., `com.buddy.core.daily.plist` in `~/Library/LaunchAgents/`) to explicitly initiate the daily sequence at 07:35 AM. Example `.plist` snippet:

      ```xml
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
          <key>Label</key>
          <string>com.buddy.core.daily</string>
          <key>ProgramArguments</key>
          <array>
              <string>/usr/bin/caffeinate</string>
              <string>-i</string>
              <string>/path/to/buddy-core/.venv/bin/python</string>
              <string>/path/to/buddy-core/src/__init__.py</string>
          </array>


        <key>WorkingDirectory</key>
        <string>/path/to/buddy-core</string>

        <key>StartCalendarInterval</key>
        <array>
            <dict>
                <key>Hour</key>
                <integer>7</integer>
                <key>Minute</key>
                <integer>35</integer>
            </dict>
        </array>

        <key>StandardOutPath</key>
        <string>/path/to/buddy-core/logs/buddy-core-cron.log</string>
        <key>StandardErrorPath</key>
        <string>/path/to/buddy-core/logs/buddy-core-cron.log</string>
      </dict>
      </plist>
      ```
