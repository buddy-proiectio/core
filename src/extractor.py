"""
The Extractor (Exact Text Extraction Agent)

This script processes gathered data using LLM-powered agents to extract structured
information, entities, and key insights. It leverages CrewAI to orchestrate
specialized extraction tasks and saves the refined results to a daily rolling
JSON file using the local timezone.
"""

import os
import json
import logging
import warnings
import re

# Suppress HuggingFace and Sentence-Transformers warnings/logs completely
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

LOG_FILE = "logs/extractor.log"

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared_logger import setup_logger

setup_logger(LOG_FILE)
logger = logging.getLogger(__name__)

from datetime import datetime
import torch
import requests

from huggingface_hub.utils import disable_progress_bars, logging as hf_hub_logging

disable_progress_bars()
hf_hub_logging.set_verbosity_error()

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer

from prompts import get_agent_config, AGENT_CONFIGS


def run_extractor(data_dir: str = None):
    """
    Executes the 'Dynamic Extraction' module for Phase 2.
    Scans for category-specific JSON files, spins up a direct LLM execution pipeline
    to perform deterministic, exact text extraction of given KPIs, and saves the result.
    """

    # --- CONFIGURATION ---
    SEMANTIC_SIMILARITY_THRESHOLD = 0.85
    # ---------------------

    # 0. Setup directories
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(data_dir, exist_ok=True)

    # 1. Get today's date in YYYYMMDD format (Local Time)
    today_str = datetime.now().strftime("%Y%m%d")

    # 2. Categories are directly driven by the configurations in prompts.py
    categories = list(AGENT_CONFIGS.keys())

    # Initialize lightweight embedding model globally for the run
    logger.info(
        "Loading embedding model for semantic deduplication (all-MiniLM-L6-v2)..."
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    active_tasks = []

    # 3. Loop through these categories
    for category in categories:
        filename = f"{category}_sorted_{today_str}.json"
        filepath = os.path.join(data_dir, filename)

        # 4. Check if the file exists
        if os.path.exists(filepath):
            # Parse JSON with error handling
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    articles = json.load(f)

                # Continue ONLY if the file contains data
                if not articles:
                    continue
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
                continue

            # Deduplicate articles by URL
            seen_urls = set()
            url_unique_articles = []
            for article in articles:
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    url_unique_articles.append(article)

            # Continue ONLY if the file contains unique data
            if not url_unique_articles:
                logger.info(f"Skipping empty/duplicate-only file: {filepath}")
                continue

            # Semantic Deduplication
            unique_articles = []
            accepted_embeddings = []
            sem_dupes_count = 0

            for article in url_unique_articles:
                title = article.get("title", "")
                content = article.get("content", "")
                # Create a fingerprint: Title + First ~3 sentences
                content_preview = " ".join(content.split(".")[:3])
                fingerprint = f"{title}. {content_preview}"

                # Encode into vector
                emb = embedder.encode(fingerprint, convert_to_tensor=True)

                is_duplicate = False
                if accepted_embeddings:
                    # Compute similarity against previously accepted articles using model.similarity
                    # which returns an N x M tensor.
                    cos_scores = embedder.similarity(
                        emb, torch.stack(accepted_embeddings)
                    )[0]
                    if cos_scores.max().item() >= SEMANTIC_SIMILARITY_THRESHOLD:
                        sem_dupes_count += 1
                        is_duplicate = True

                if not is_duplicate:
                    unique_articles.append(article)
                    accepted_embeddings.append(emb)

            if sem_dupes_count > 0:
                logger.info(
                    f"Filtered {sem_dupes_count} semantic duplicates (Threshold: {SEMANTIC_SIMILARITY_THRESHOLD}) in category: {category}"
                )
            else:
                logger.info(f"No semantic duplicates found in category: {category}")

            if not unique_articles:
                logger.info(
                    f"No unique articles left after semantic deduplication: {filepath}"
                )
                continue

            # Fetch the dynamic configuration for this role
            config = get_agent_config(category)
            keywords = config.get("keywords", [])
            # Compile regex patterns for exact word boundary matches
            keyword_patterns = (
                [re.compile(r"\b" + re.escape(kw.lower()) + r"\b") for kw in keywords]
                if keywords
                else []
            )

            # Build the strict System Prompt
            system_prompt = (
                f"Role: {config['role']}\n\n"
                f"Goal: {config['goal']}\n\n"
                f"Backstory: {config['backstory']}\n\n"
                f"CRITICAL RULE: DO NOT WRAP YOUR RESPONSE IN 'Thought:', 'Action:', or any other conversational filler. ONLY RETURN THE EXACT TEXT OR 'NO_EXTRACTION'."
            )

            # Format input data for the Task and create 1-to-1 Task Mapping
            for idx, article in enumerate(unique_articles):
                title = article.get("title", f"Article {idx+1}")
                content = article.get("content", "")

                # 1st Pass: Python Keyword Pre-filtering
                if keyword_patterns:
                    combined_text = (title + " " + content).lower()
                    if not any(
                        pattern.search(combined_text) for pattern in keyword_patterns
                    ):
                        # Skip this article completely if no keywords match
                        continue

                input_text = (
                    f"\n--- BEGIN ARTICLE ---\n"
                    f"Title: {title}\n"
                    f"Content:\n{content}\n"
                    f"--- END ARTICLE ---\n"
                )

                # Construct task description via template injection
                task_description = config["task_description_template"].format(
                    input_text=input_text
                )

                active_tasks.append(
                    (category, system_prompt, task_description, article)
                )

            logger.info(
                f"Added {len(unique_articles)} Extraction Tasks for category: {category}"
            )
        else:
            logger.info(f"File not found, skipping category: {category}")

    # Process all tasks directly via Local LLM
    if active_tasks:
        logger.info(f"Starting Direct LLM Extraction with {len(active_tasks)} tasks...")
        logger.info(
            "Bypassing Agent framework overhead (ReAct loop) to maximize speed!"
        )

        output_filename = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

        # Group task outputs by category for structured writing
        category_outputs = {category: [] for category in categories}

        try:
            url = "http://127.0.0.1:11434/api/chat"
            total_tasks = len(active_tasks)

            # Setup TCP connection pool session
            with requests.Session() as session:
                for idx, task_data in enumerate(active_tasks, 1):
                    category, sys_prompt, user_prompt, article = task_data

                    payload = {
                        "model": "llama3.1",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {
                            "num_ctx": 4096,
                            "num_predict": 1500,
                            "temperature": 0.0,
                            "top_p": 0.1,
                        },
                    }

                    try:
                        logger.info(
                            f"Processing Task {idx:02d}/{total_tasks} [{category}] ..."
                        )
                        # Synchronous post with TCP connection pooling
                        resp = session.post(url, json=payload, timeout=600)
                        resp.raise_for_status()
                        data = resp.json()
                        output = data.get("message", {}).get("content", "").strip()

                        # Rigid post-processing block
                        if output and "NO_EXTRACTION" not in output.upper():
                            logger.info(f"Task {idx:02d} [{category}] Extracted!")
                            # Replace all newlines with a single space to form a continuous block
                            output = re.sub(r"\s+", " ", output).strip()

                            # Format the result with python
                            raw_title = article.get("title", f"Article {idx}")
                            # Strip HTML tags
                            clean_title = re.sub(r"<.*?>", "", raw_title).strip()
                            article_url = article.get("url", "#")

                            # Dedup check: If output text is just the title, printing it repeats the title.
                            if output == clean_title:
                                final_text = f"[{clean_title}]({article_url})"
                            else:
                                final_text = f"[{clean_title}]({article_url})\n{output}"

                            category_outputs[category].append(final_text)
                        else:
                            logger.info(
                                f"Task {idx:02d} [{category}] Skipped (No KPIs)"
                            )

                    except Exception as e:
                        logger.error(f"Error process Task {idx:02d} [{category}]: {e}")

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Process terminating.")
            logger.info("Saving partially extracted progress before exiting...")

        # Save output to text file with filtering
        with open(output_filename, "w", encoding="utf-8") as out_f:
            cnt = 0
            for cat, outputs in category_outputs.items():
                if outputs:
                    cnt += 1
                    out_f.write(f"### {cat}\n\n")
                    for out in outputs:
                        out_f.write(f"{out}\n\n")

        if cnt == 0:
            logger.info(
                f"Extraction finished, but NO facts were found. Categories written: 0."
            )
        else:
            logger.info(
                f"Extraction complete! Unique, filtered output saved to: {output_filename}"
            )

        return True
    else:
        logger.info(
            f"No active tasks were created for date {today_str}. Extraction aborted."
        )
        return None
