"""
The Extractor (Exact Text Extraction Agent)

This script processes gathered data using LLM-powered agents to extract structured
information, entities, and key insights. It leverages CrewAI to orchestrate
specialized extraction tasks and saves the refined results to a daily rolling
JSON file using the local timezone.
"""

import os
import json
from datetime import datetime
from litellm import completion

from prompts import get_agent_config, AGENT_CONFIGS


def run_extraction_pipeline(data_dir: str = "."):
    """
    Executes the 'Dynamic Extraction' module for Phase 2.
    Scans for category-specific JSON files, spins up a direct LLM execution pipeline
    to perform deterministic, exact text extraction of given KPIs, and saves the result.
    """

    # 1. Get today's date in YYYYMMDD format (Local Time)
    today_str = datetime.now().strftime("%Y%m%d")

    # 2. Categories are directly driven by the configurations in prompts.py
    categories = list(AGENT_CONFIGS.keys())

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
                print(f"Error reading {filepath}: {e}")
                continue

            # Deduplicate articles by URL
            seen_urls = set()
            unique_articles = []
            for article in articles:
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)

            # Continue ONLY if the file contains unique data
            if not unique_articles:
                print(f"Skipping empty/duplicate-only file: {filepath}")
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
                title = article.get("title", f"Article {idx+1}")
                url = article.get("url", "#")
                content = article.get("content", "")

                input_text = (
                    f"\n--- BEGIN ARTICLE ---\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content:\n{content}\n"
                    f"--- END ARTICLE ---\n"
                )

                # Construct task description via template injection
                task_description = config["task_description_template"].format(
                    input_text=input_text
                )

                active_tasks.append((category, system_prompt, task_description))

            print(
                f"Added {len(unique_articles)} Extraction Tasks for category: {category}"
            )
        else:
            print(f"File not found, skipping category: {category}")

    # Process all tasks directly via LLM
    if active_tasks:
        print(f"\nStarting Direct LLM Extraction with {len(active_tasks)} tasks...")
        print("Bypassing Agent framework overhead (ReAct loop) to maximize speed!\n")

        output_filename = os.path.join(data_dir, f"extracted_facts_{today_str}.txt")

        # Group task outputs by category for structured writing
        category_outputs = {category: [] for category in categories}

        for idx, (category, sys_prompt, user_prompt) in enumerate(active_tasks, 1):
            print(
                f"Processing Task {idx:02d}/{len(active_tasks)} [{category}] ... ",
                end="",
                flush=True,
            )
            try:
                response = completion(
                    model="ollama/llama3.1",
                    api_base="http://localhost:11434",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,  # Deterministic, limits hallucination depth
                    max_tokens=600,  # Sufficient for pure text block extraction
                )

                output = response.choices[0].message.content.strip()

                # Rigid post-processing block
                if output and "NO_EXTRACTION" not in output.upper():
                    print("Extracted!")
                    category_outputs[category].append(output)
                else:
                    print("Skipped (No KPIs)")

            except Exception as e:
                print(f"Error process: {e}")

        # Save output to text file with filtering
        with open(output_filename, "w", encoding="utf-8") as out_f:
            cnt = 0
            for category, outputs in category_outputs.items():
                if outputs:
                    cnt += 1
                    out_f.write(f"### {category}\n\n")
                    for out in outputs:
                        out_f.write(f"{out}\n\n")

        if cnt == 0:
            print(
                f"\nExtraction finished, but NO facts were found. Categories written: 0."
            )
        else:
            print(
                f"\nExtraction complete! Unique, filtered output saved to: {output_filename}"
            )

        return True
    else:
        print(
            f"\nNo active tasks were created for date {today_str}. Extraction aborted."
        )
        return None


if __name__ == "__main__":
    # Note: Ensure you run this from the 'extractor' directory or adjust data_dir
    run_extraction_pipeline()
