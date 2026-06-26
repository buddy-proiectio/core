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
import pytz
import sys
import argparse
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import torch
import typing
import requests
from huggingface_hub.utils import disable_progress_bars, logging as hf_hub_logging
from sentence_transformers import SentenceTransformer

from shared.shared_logger import setup_logger

from prompts import get_agent_config, AGENT_CONFIGS

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

logger = setup_logger(LOG_FILE, __name__)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

disable_progress_bars()
hf_hub_logging.set_verbosity_error()


def strip_captions(text: str) -> str:
    """
    Strips out lines containing image, chart, or screenshot source captions
    as well as standalone chart labels, infographics, and diagram titles
    to prevent them from being processed as facts.
    """
    if not text:
        return ""

    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append(line)
            continue

        lower_line = stripped.lower()

        # 1. If line contains "Source:" (case-insensitive) and looks like a caption
        if re.search(r"\bSource\s*:", stripped, re.IGNORECASE):
            if any(
                k in lower_line
                for k in [
                    "chart",
                    "screenshot",
                    "photo",
                    "image",
                    "diagram",
                    "source:",
                    "coinglass",
                    "tradingview",
                    "cryptic",
                    "cointelegraph",
                    "glassnode",
                    "cryptoquant",
                ]
            ):
                logger.info(f"Stripping caption line: {stripped}")
                continue

        # 2. Strip standalone chart titles, consensus-charts, infographic labels, or quote buttons/links
        is_chart_caption = False

        # If it's a markdown link [Text](URL) where the Text or URL is related to a chart, infographic, or diagram
        # e.g., [Chart of the Day](...) or [Nvidia vs Intel: diverging paths](.../charts/...)
        link_match = re.match(r"^\[(.*?)\]\((.*?)\)(?:\s*<br\s*/?>)?$", stripped)
        if link_match:
            link_title = link_match.group(1).lower()
            link_url = link_match.group(2).lower()
            if any(
                k in link_title
                for k in [
                    "chart",
                    "infographic",
                    "diagram",
                    "table",
                    "consensus-chart",
                    "price-consensus",
                ]
            ):
                is_chart_caption = True
            elif any(
                k in link_url for k in ["chart", "infographic", "diagram", "table"]
            ):
                is_chart_caption = True

        # If it contains "price-consensus-chart" or "price-consensus" or "consensus-chart" or "price and consensus" or "price & consensus"
        if any(
            k in lower_line
            for k in [
                "price-consensus",
                "consensus-chart",
                "price and consensus",
                "price & consensus",
            ]
        ):
            is_chart_caption = True

        # If it has the quote buttons/links typical of financial sites like Zacks or Yahoo Finance
        # e.g. "Super Micro Computer, Inc. Quote" or "SMCI Quote" or "Alphabet Inc. Quote"
        if re.search(r"\|\s*[\w\s.,&-]+\s+quote\b", lower_line):
            is_chart_caption = True
        elif lower_line.endswith("quote") and len(stripped) < 60:
            is_chart_caption = True

        # Standalone short lines ending with or containing chart/charts/infographic/infographics/diagram/diagrams
        # e.g. "YTD Performance Chart", "Forward 12 Month (P/S) Valuation Chart"
        if len(stripped) < 120:
            # Matches words like chart, charts, infographic, infographics, diagram, diagrams
            # Or table followed by a number/letter (e.g. "Table 1", "Table A")
            if re.search(
                r"\b(chart|charts|infographic|infographics|diagram|diagrams)\b",
                lower_line,
            ) or re.search(r"\btable\s+(\d+|[a-gi-z]\b)", lower_line):
                # Ensure it's not a normal sentence like "The FED left rates unchanged, as shown in the chart."
                # Normal sentences have verbs, lowercase words, are punctuated, etc.
                if (
                    not stripped.endswith(".")
                    or len(stripped) < 60
                    or "performance chart" in lower_line
                    or "valuation chart" in lower_line
                ):
                    is_chart_caption = True

        if is_chart_caption:
            logger.info(f"Stripping standalone chart/infographic caption: {stripped}")
            continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def strip_tables(text: str) -> str:
    """
    Strips out markdown tables and plaintext financial table/statement blocks from the text
    to prevent them from being processed by the LLM.
    """
    if not text:
        return ""

    # Split by lines
    lines = text.split("\n")
    filtered_lines = []

    # Block-start keywords (case-insensitive) for plaintext financial statements/tables
    table_block_headers = [
        "consolidated statements of current earnings",
        "consolidated statements of earnings",
        "consolidated statements of operations",
        "consolidated statements of comprehensive income",
        "consolidated balance sheets",
        "consolidated statements of cash flows",
        "condensed consolidated statements",
        "condensed consolidated balance sheets",
        "non-gaap financial measure reconciliation",
        "non-gaap financial measures reconciliation",
        "reconciliation of adjusted operating",
        "reconciliation of adjusted diluted",
        "condensed consolidated statements of earnings",
        "condensed consolidated statements of cash flows",
        "three months ended",
    ]

    in_table_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append(line)
            continue

        # --- Markdown Table Row Detection ---
        is_md_table_line = False
        if "|" in stripped:
            if stripped.count("|") >= 2:
                is_md_table_line = True
            elif stripped.startswith("|") or stripped.endswith("|"):
                is_md_table_line = True
            elif stripped == "|":
                is_md_table_line = True

        # Separator row checking (e.g. |---| or :---:)
        if not is_md_table_line:
            if (
                set(stripped).issubset({"|", "-", ":", " "})
                and "-" in stripped
                and "|" in stripped
            ):
                is_md_table_line = True

        if is_md_table_line:
            continue

        # --- Plaintext Financial Statement Block Detection ---
        lower_stripped = stripped.lower()
        should_start_block = False
        for header in table_block_headers:
            if header in lower_stripped:
                # Special rule for "three months ended": only trigger if the line is short (table header)
                if header == "three months ended":
                    if len(stripped) < 60:
                        should_start_block = True
                else:
                    should_start_block = True
                break

        if should_start_block:
            in_table_block = True
            continue

        if in_table_block:
            # Check if this line looks like a normal narrative paragraph to exit the block
            digit_count = sum(c.isdigit() for c in stripped)
            digit_ratio = digit_count / len(stripped) if stripped else 0

            is_exit_paragraph = False
            if len(stripped) >= 100 and digit_ratio < 0.15 and stripped.endswith("."):
                is_exit_paragraph = True
            elif stripped.lower().startswith("about ") and len(stripped) < 50:
                is_exit_paragraph = True

            if is_exit_paragraph:
                in_table_block = False
            else:
                continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def is_summary_or_hallucination(text: str) -> bool:
    """
    Checks if the output looks like a summary, conversational filler, prompt leakage,
    or self-referential description rather than direct exact sentence extraction.
    """
    if not text:
        return False

    text_lower = text.lower().strip()

    # 1. Prompt leakage / instruction echoing (Check anywhere in the text)
    leakage_patterns = [
        r"extraction\s+session\s+salt",
        r"session\s+salt",
        r"extraction\s+session",
        r"zero-creativity\s+extraction",
        r"gating\s+rule",
        r"kpi\s+extraction",
        r"100%\s+exact\s+substring\s+match",
        r"output\s+format",
        r"chronological\s+flow",
        r"clutter\s+filtering",
    ]
    for pattern in leakage_patterns:
        if re.search(pattern, text_lower):
            return True

    # 2. Conversational fillers, safety refusals, and summary intros (Check starting patterns)
    prefix_patterns = [
        r"^(this|the|in\s+this|in\s+the)\s+(article|text|post|news|excerpt|report|piece|document|summary|press\s+release)\b",
        r"^(here\s+is|here\s+are)\b",
        r"^summary\b",
        r"^(i\s+have|i\s+will|i\s+shall|i\s+cannot|i\s+am\s+unable|i\s+apologize|i\s+do\s+not\s+have|i\s+am\s+not\s+programmed)\b",
        r"^extracted\s+(sentence|kpi|fact|text)\b",
        r"^based\s+on\s+the\s+(article|text|post|news|excerpt|report|piece|document|rules|instructions|goals)\b",
        r"^according\s+to\s+the\s+(article|text|post|news|excerpt|report|piece|document|rules|instructions|goals)\b",
        r"^here\s+is\s+the\s+extraction\b",
        r"^here\s+is\s+what\s+i\s+extracted\b",
        r"^extracted\s+from\s+the\s+(article|text)\b",
        r"^no\s+(relevant\s+)?(kpis|facts|data|sentences|information)\b",
        r"^there\s+(is|are)\s+no\s+(relevant\s+)?(kpis|facts|data|sentences|information)\b",
        r"^this\s+article\s+does\s+not\b",
        r"^the\s+provided\s+text\s+does\s+not\b",
    ]

    for pattern in prefix_patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def verify_exact_match(output: str, original_content: str) -> bool:
    """
    Verifies that every sentence in the LLM output is an exact substring of the original article content.
    """
    if not output or not original_content:
        return False

    # Clean and pre-process original content
    content_clean = strip_tables(original_content)
    content_clean = strip_captions(content_clean)
    content_clean = " ".join(content_clean.split()).lower()

    # Split output into sentences using regex
    sentences = re.split(r"(?<=[.!?])\s+", output)

    for sentence in sentences:
        sentence_clean = " ".join(sentence.split()).strip().lower()
        if not sentence_clean:
            continue

        # If it's a special token like NO_EXTRACTION, skip it
        if "no_extraction" in sentence_clean or "no data" in sentence_clean:
            continue

        if sentence_clean not in content_clean:
            # Fallback check stripping punctuation/symbols
            sentence_stripped = re.sub(r"[^\w\s]", "", sentence_clean)
            content_stripped = re.sub(r"[^\w\s]", "", content_clean)
            if sentence_stripped not in content_stripped:
                logger.warning(
                    f"Exact match validation failed for sentence: '{sentence_clean[:60]}...'"
                )
                return False

    return True


def run_extractor(
    data_dir: typing.Optional[str] = None,
    report_type: typing.Optional[str] = None,
) -> typing.Optional[bool]:
    """
    Executes the 'Dynamic Extraction' module for Phase 2.
    Scans for category-specific JSON files, spins up a direct LLM execution pipeline
    to perform deterministic, exact text extraction of given KPIs, and saves the result.
    """

    # --- CONFIGURATION ---
    SEMANTIC_SIMILARITY_THRESHOLD = 0.82
    # ---------------------

    # 0. Setup directories
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(data_dir, exist_ok=True)

    # 1. Get today's date in YYYYMMDD format (US Eastern Time)
    us_tz = pytz.timezone("America/New_York")
    today_str = datetime.now(us_tz).strftime("%Y%m%d")

    # 2. Categories are directly driven by the configurations in prompts.py
    categories = list(AGENT_CONFIGS.keys())

    # 2.5 Load premarket cache if this is not a premarket run
    pre_state_data = None
    if report_type != "premarket":
        pre_state_filename = os.path.join(data_dir, "extracted_state_pre.json")
        if os.path.exists(pre_state_filename):
            try:
                with open(pre_state_filename, "r", encoding="utf-8") as f:
                    pre_state_data = json.load(f)
                logger.info(
                    "Loaded premarket state cache from extracted_state_pre.json"
                )
            except Exception as e:
                logger.error(f"Failed to load premarket state cache: {e}")

    # 2.6 Load incremental state
    state_filename = os.path.join(data_dir, f"extracted_state_{today_str}.json")
    state_data: dict[str, typing.Any] = {
        "extracted_urls": [],
        "category_normal_outputs": {cat: [] for cat in categories},
        "category_sec_outputs": {cat: [] for cat in categories},
    }
    if os.path.exists(state_filename):
        try:
            with open(state_filename, "r", encoding="utf-8") as f:
                loaded_state = json.load(f)
                if "extracted_urls" in loaded_state:
                    state_data = loaded_state
                logger.info(
                    f"Loaded incremental state with {len(state_data['extracted_urls'])} previously processed articles."
                )
        except Exception as e:
            logger.error(f"Failed to load state file {state_filename}: {e}")

    # 2.7 Merge premarket state into state_data if present
    if pre_state_data:
        # Merge URLs
        for url in pre_state_data.get("extracted_urls", []):
            if url not in state_data["extracted_urls"]:
                state_data["extracted_urls"].append(url)

        # Merge category normal outputs
        pre_normal = pre_state_data.get("category_normal_outputs", {})
        for cat, outputs in pre_normal.items():
            if cat in state_data["category_normal_outputs"]:
                for out in outputs:
                    if out not in state_data["category_normal_outputs"][cat]:
                        state_data["category_normal_outputs"][cat].append(out)

        # Merge category sec outputs
        pre_sec = pre_state_data.get("category_sec_outputs", {})
        for cat, outputs in pre_sec.items():
            if cat in state_data["category_sec_outputs"]:
                for out in outputs:
                    if out not in state_data["category_sec_outputs"][cat]:
                        state_data["category_sec_outputs"][cat].append(out)

        logger.info(
            f"Merged premarket cache: now has {len(state_data['extracted_urls'])} URLs."
        )

        # Delete premarket cache file after successful merge so it's not merged again
        try:
            pre_state_filename = os.path.join(data_dir, "extracted_state_pre.json")
            os.remove(pre_state_filename)
            logger.info("Removed merged premarket cache file extracted_state_pre.json")
        except Exception as e:
            logger.error(f"Failed to remove premarket cache file: {e}")

    extracted_urls_set = set(state_data["extracted_urls"])

    extracted_urls: list[str] = typing.cast(list[str], state_data["extracted_urls"])
    if "category_sec_outputs" not in state_data:
        state_data["category_sec_outputs"] = {}
    if "category_normal_outputs" not in state_data:
        state_data["category_normal_outputs"] = {}

    category_sec_outputs: dict[str, list[str]] = typing.cast(
        dict[str, list[str]], state_data["category_sec_outputs"]
    )
    category_normal_outputs: dict[str, list[str]] = typing.cast(
        dict[str, list[str]], state_data["category_normal_outputs"]
    )

    # Ensure newly added categories exist in the state dictionaries
    for cat in categories:
        if cat not in category_sec_outputs:
            category_sec_outputs[cat] = []
        if cat not in category_normal_outputs:
            category_normal_outputs[cat] = []

    # Initialize lightweight embedding model globally for the run
    logger.info(
        "Loading embedding model for semantic deduplication (all-MiniLM-L6-v2)..."
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    active_tasks = []

    # 3. Loop through these categories
    for category in categories:
        safe_category = category.replace(" ", "_").replace("/", "_")
        filename = f"{safe_category}_sorted_{today_str}.json"
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
                    # Skip if already extracted in a previous run today
                    if url not in extracted_urls_set:
                        url_unique_articles.append(article)

            # Continue ONLY if the file contains unique data
            if not url_unique_articles:
                logger.info(f"Skipping empty/duplicate-only file: {filepath}")
                continue

            # Semantic Deduplication
            unique_articles = []
            accepted_embeddings: list[typing.Any] = []
            sem_dupes_count = 0

            for article in url_unique_articles:
                article_url = article.get("url", "")
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
                    emb_tensor: typing.Any = emb
                    accepted_tensor = torch.stack(accepted_embeddings)  # type: ignore
                    embedder_any: typing.Any = embedder
                    cos_scores = embedder_any.similarity(emb_tensor, accepted_tensor)[0]
                    if cos_scores.max().item() >= SEMANTIC_SIMILARITY_THRESHOLD:
                        sem_dupes_count += 1
                        is_duplicate = True

                if is_duplicate:
                    # Add to state so we don't re-check it in future incremental runs
                    extracted_urls_set.add(article_url)
                    extracted_urls.append(article_url)
                    continue

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

            # Build the strict System Prompt
            system_prompt = (
                f"Role: {config['role']}\n\n"
                f"Goal: {config['goal']}\n\n"
                f"Backstory: {config['backstory']}\n\n"
                f"CRITICAL RULE: DO NOT WRAP YOUR RESPONSE IN 'Thought:', 'Action:', or any other conversational filler. ONLY RETURN THE EXACT TEXT OR 'NO_EXTRACTION'."
            )

            # Format input data for the Task and create 1-to-1 Task Mapping
            for idx, article in enumerate(unique_articles):
                title = article.get("title", f"Article {idx + 1}")
                content = article.get("content", "")

                # Pre-filter out tables (markdown & financial lists),
                # strip out image captions, remove emojis from input to save tokens and ensure clean extraction
                content = strip_tables(content)
                content = strip_captions(content)
                title = re.sub(r"[\U00010000-\U0010ffff]", "", title)
                content = re.sub(r"[\U00010000-\U0010ffff]", "", content)

                input_text = (
                    f"\n--- BEGIN ARTICLE ---\n"
                    f"Title: {title}\n"
                    f"Content:\n{content}\n"
                    f"--- END ARTICLE ---\n"
                )

                # Inject a unique salt to bust Ollama prefix caching and force fresh context isolation
                cache_buster = f"Extraction Session Salt: {uuid.uuid4().hex}\n"
                task_description = cache_buster + config[
                    "task_description_template"
                ].format(input_text=input_text)

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

        # Use the state loaded earlier
        category_normal_outputs = typing.cast(
            dict[str, list[str]], state_data["category_normal_outputs"]
        )
        category_sec_outputs = typing.cast(
            dict[str, list[str]], state_data["category_sec_outputs"]
        )

        try:
            url = "http://127.0.0.1:11434/api/chat"
            total_tasks = len(active_tasks)

            # Setup TCP connection pool session
            with requests.Session() as session:
                for idx, task_data in enumerate(active_tasks, 1):
                    category, sys_prompt, user_prompt, article = task_data

                    # SEC Filing Bypass: Skip LLM completely and just output the formatted title
                    if article.get("extraction_status") == "sec_filing":
                        raw_title = article.get("title", f"Article {idx}")
                        clean_title = re.sub(r"<.*?>", "", raw_title).strip()
                        clean_title = re.sub(
                            r"[\U00010000-\U0010ffff]", "", clean_title
                        )
                        article_url = article.get("url", "#")

                        logger.info(
                            f"Task {idx:02d}/{total_tasks} [{category}] SEC Filing detected. Bypassing LLM."
                        )
                        final_text = f"[{clean_title}]({article_url})"

                        # Double-check duplicates by URL or title
                        is_dup = False
                        for existing in category_sec_outputs[category]:
                            if article_url != "#" and (
                                article_url in existing or clean_title in existing
                            ):
                                is_dup = True
                                break
                        if not is_dup:
                            for existing in category_normal_outputs[category]:
                                if article_url != "#" and (
                                    article_url in existing or clean_title in existing
                                ):
                                    is_dup = True
                                    break

                        if not is_dup:
                            category_sec_outputs[category].append(final_text)
                        continue

                    max_attempts = 3
                    attempt = 0
                    success = False
                    output = ""
                    current_user_prompt = user_prompt

                    while attempt < max_attempts and not success:
                        attempt += 1

                        temp = 0.0
                        top_p = 0.1
                        if attempt == 2:
                            temp = 0.2
                            top_p = 0.2
                            current_user_prompt = user_prompt + (
                                "\n\n[WARNING - RETRY] Your previous response was rejected because you summarized the article, used self-referential conversational filler, or hallucinated facts not present in the text. "
                                "DO NOT summarize the article. DO NOT describe it. DO NOT start with 'This article...'. "
                                "You must ONLY copy and paste the exact sentences from the source text that contain the KPIs. "
                                "If no KPIs are present, reply with 'NO_EXTRACTION'."
                            )
                        elif attempt == 3:
                            temp = 0.4
                            top_p = 0.3
                            current_user_prompt = user_prompt + (
                                "\n\n[CRITICAL WARNING - FINAL RETRY] Do NOT start your response with 'This article...', 'The article...', 'In this article...', or any summary description. "
                                "Do NOT hallucinate or output facts not present in the text. You are a deterministic copy-paste engine. "
                                "Directly output the exact sentence(s) from the text, or output 'NO_EXTRACTION'."
                            )

                        payload = {
                            "model": OLLAMA_MODEL,
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": current_user_prompt},
                            ],
                            "stream": False,
                            "options": {
                                "num_ctx": 8192,
                                "num_predict": 1500,
                                "temperature": temp,
                                "top_p": top_p,
                                "num_keep": 0,
                            },
                        }

                        try:
                            if attempt == 1:
                                logger.info(
                                    f"Processing Task {idx:02d}/{total_tasks} [{category}] ..."
                                )
                            else:
                                logger.info(
                                    f"Retrying Task {idx:02d}/{total_tasks} [{category}] (Attempt {attempt}/{max_attempts}) due to summary/filler/hallucination detection..."
                                )

                            # Synchronous post with TCP connection pooling
                            resp = session.post(url, json=payload, timeout=600)
                            resp.raise_for_status()
                            data = resp.json()
                            raw_output = (
                                data.get("message", {}).get("content", "").strip()
                            )

                            if not raw_output:
                                output = ""
                                success = True
                                break

                            # Check if the output is a summary or has conversational filler
                            if is_summary_or_hallucination(raw_output):
                                logger.warning(
                                    f"Task {idx:02d} [{category}] Attempt {attempt} output matched summary/filler pattern: '{raw_output[:80]}...'"
                                )
                                continue

                            # Verify that the extracted text is actually present in the original article
                            if not verify_exact_match(
                                raw_output, article.get("content", "")
                            ):
                                logger.warning(
                                    f"Task {idx:02d} [{category}] Attempt {attempt} output failed exact match validation (hallucination/leak detected)."
                                )
                                continue

                            output = raw_output
                            success = True

                        except Exception as e:
                            logger.error(
                                f"Error processing Task {idx:02d} [{category}] on attempt {attempt}: {e}"
                            )
                            if attempt == max_attempts:
                                break

                    try:
                        # Rigid post-processing block
                        if output:
                            # Squash all newlines and excess whitespace natively
                            output = " ".join(output.split())

                            # Remove pic.twitter.com links
                            output = re.sub(
                                r"\s*(https?://)?pic\.twitter\.com/[A-Za-z0-9_/-]+",
                                "",
                                output,
                            )

                            # Remove Source captions from output (e.g., "... Source: Cointelegraph/TradingView")
                            output = re.sub(
                                r"(?i)\bSource\s*:\s*[A-Za-z0-9_/-]+(?:/[A-Za-z0-9_/-]+)*(?:\s*/[A-Za-z0-9_/-]+)?",
                                "",
                                output,
                            )

                            # Escape markdown numbered lists (only at the start of a line) to prevent automatic formatting
                            output = re.sub(
                                r"^(\s*)(\d+)\.(\s)",
                                r"\1\2\.\3",
                                output,
                                flags=re.MULTILINE,
                            )

                            # Final emoji removal from LLM output
                            output = re.sub(r"[\U00010000-\U0010ffff]", "", output)
                            # Remove miscellaneous symbols like ⚠️
                            output = re.sub(
                                r"[\u2600-\u27BF\u2300-\u23FF\u2B50\u2B55]", "", output
                            )

                            # Format the result with python
                            raw_title = article.get("title", f"Article {idx}")
                            # Strip HTML tags and emojis
                            clean_title = re.sub(r"<.*?>", "", raw_title).strip()
                            clean_title = re.sub(
                                r"[\U00010000-\U0010ffff]", "", clean_title
                            )
                            article_url = article.get("url", "#")

                            upper_output = output.upper()
                            first_line_upper = upper_output.split("\n")[0].strip()
                            words = output.split()
                            letters_count = sum(c.isalpha() for c in output)
                            digits_count = sum(c.isdigit() for c in output)

                            # 1. Skip completely if it's a garbage article (NO_EXTRACTION) or mostly numbers
                            if (
                                "NO_EXTRACTION" in first_line_upper
                                or "NO DATA" in first_line_upper
                            ):
                                logger.info(
                                    f"Task {idx:02d} [{category}] NO_EXTRACTION detected on first line. Skipping article completely."
                                )
                                extracted_urls_set.add(article_url)
                                extracted_urls.append(article_url)
                                continue

                            if letters_count == 0 or (
                                digits_count > letters_count * 0.8
                            ):
                                logger.info(
                                    f"Task {idx:02d} [{category}] Extracted mostly numbers. Skipping article completely."
                                )
                                extracted_urls_set.add(article_url)
                                extracted_urls.append(article_url)
                                continue

                            # 2. Heuristic for "contextless numbers" (e.g., "$10.6 billion $2.65 38%")
                            # If a high percentage of tokens are numbers and there's no structural punctuation, it's a data dump without context.
                            number_tokens = sum(
                                1 for w in words if any(c.isdigit() for c in w)
                            )
                            number_ratio = number_tokens / len(words) if words else 0
                            has_punctuation = bool(
                                re.search(r"[,:;?!]|\.\s|\.$", output)
                            )
                            is_contextless = (
                                number_ratio >= 0.30 and not has_punctuation
                            )

                            # 3. Fallback to Title-only if extraction is very short, matches title, or is contextless
                            if (
                                len(words) < 10
                                or output == clean_title
                                or is_contextless
                            ):
                                reason = (
                                    "Short extraction or matches title"
                                    if not is_contextless
                                    else "Extracted contextless amounts"
                                )
                                logger.info(
                                    f"Task {idx:02d} [{category}] {reason}. Fallback to Title-only."
                                )
                                final_text = f"[{clean_title}]({article_url})"
                            else:
                                logger.info(f"Task {idx:02d} [{category}] Extracted!")
                                final_text = f"[{clean_title}]({article_url})\n{output}"

                            # Handle cases where SEC filings slip into normal extraction
                            is_dup = False
                            for existing in category_sec_outputs[category]:
                                if article_url != "#" and (
                                    article_url in existing or clean_title in existing
                                ):
                                    is_dup = True
                                    break
                            if not is_dup:
                                for existing in category_normal_outputs[category]:
                                    if article_url != "#" and (
                                        article_url in existing
                                        or clean_title in existing
                                    ):
                                        is_dup = True
                                        break

                            if not is_dup:
                                if re.search(
                                    r"\b(8-K|10-K|10-Q|SEC Filing)\b",
                                    clean_title,
                                    re.IGNORECASE,
                                ):
                                    category_sec_outputs[category].append(final_text)
                                else:
                                    category_normal_outputs[category].append(final_text)

                            # Mark as extracted
                            extracted_urls_set.add(article_url)
                            extracted_urls.append(article_url)
                        else:
                            logger.info(
                                f"Task {idx:02d} [{category}] Skipped (Empty output)"
                            )

                    except Exception as e:
                        logger.error(f"Error process Task {idx:02d} [{category}]: {e}")

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Process terminating.")
            logger.info("Saving partially extracted progress before exiting...")

        # Save the updated state
        try:
            with open(state_filename, "w", encoding="utf-8") as sf:
                json.dump(state_data, sf, ensure_ascii=False, indent=2)

            # If this is a premarket run, also save a copy to extracted_state_pre.json
            if report_type == "premarket":
                pre_state_filename = os.path.join(data_dir, "extracted_state_pre.json")
                with open(pre_state_filename, "w", encoding="utf-8") as psf:
                    json.dump(state_data, psf, ensure_ascii=False, indent=2)
                logger.info("Saved premarket cache to extracted_state_pre.json")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

        # Save output to text file with filtering
        with open(output_filename, "w", encoding="utf-8") as out_f:
            total_facts = 0
            for cat in categories:
                sec_outputs = category_sec_outputs.get(cat, [])
                normal_outputs = category_normal_outputs.get(cat, [])

                out_f.write(f"### {cat}\n\n")

                if sec_outputs:
                    # Sort SEC filings alphabetically
                    sec_outputs.sort(key=lambda x: x.upper())
                    for out in sec_outputs:
                        out_f.write(f"{out}\n\n")
                        total_facts += 1

                if normal_outputs:
                    for out in normal_outputs:
                        out_f.write(f"{out}\n\n")
                        total_facts += 1

        if total_facts == 0:
            logger.info("Extraction finished, but NO facts were found.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extractor Agent")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the sorted JSON files",
    )
    parser.add_argument(
        "--report-type",
        type=str,
        default=None,
        help="Type of the report pipeline (premarket, incremental, full)",
    )
    args = parser.parse_args()
    run_extractor(data_dir=args.data_dir, report_type=args.report_type)
