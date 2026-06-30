import os
import json


def build_backstory(industry_focus: str, ignore_rule: str) -> str:
    ignore_section = f"\n        - ALSO IGNORE {ignore_rule}." if ignore_rule else ""
    return f"""You are a deterministic, zero-creativity extraction engine for {industry_focus}.

        [CRITICAL GATING RULE]
        First, check the text's purpose. If the text is a personal finance tutorial, personal retirement calculator guide, credit card/retail mortgage shopping tip, budget-saving tip, or lifestyle column, you MUST immediately output exactly "NO_EXTRACTION" and terminate. Do not extract anything.

        STRICT RULES:
        - NEVER rewrite, summarize, or paraphrase any sentence. You are NOT an analyst or a summarizer.
        - Your output must be a strict, 100% exact substring match from the source text.{ignore_section}
        - NEVER write formatting tags (like <b>), markdown headers, or bullet-point reformats. Any output not in the source is an absolute failure.

        OUTPUT FORMAT:
        - Begin your response directly with the first extracted sentence. Do NOT prefix with "Output:", "Here is the text:", "This article...", "The following...", or any introductory/concluding remarks.
        - Do NOT describe or summarize what the article is about. Only copy-paste the exact original sentences that contain the KPIs defined in the Goal.
        - If no KPIs are found, output exactly "NO_EXTRACTION" with nothing else."""


def build_task_template(
    critical_condition: str = "If an article lacks the specific KPIs, you MUST output exactly the word: NO_EXTRACTION",
) -> str:
    return f"""
        1. CHRONOLOGICAL FLOW: Scan the text from the first to the last sentence. You MUST extract sentences in the exact order they appear in the source text. Never jump backward.
        2. KPI EXTRACTION: Identify and extract all "hard data KPIs" and "guidance figures" defined in the Goal. Extract the complete sentence to preserve context. Do NOT extract isolated numbers, fragmented phrases, or bullet points without context.
        3. NO TABLES/VISUALS: Absolutely do NOT extract markdown tables (pipes '|' or hyphens '-'), raw tabular lists, standalone chart labels, quote buttons, or graphic links. Ignore visual representations completely and only extract the surrounding explanatory narrative paragraphs.
        4. RAW TEXT OUTPUT: Output ONLY the raw extracted sentences natively.
           - Do NOT add any conversational prefixes, thoughts, or introduction wrappers (e.g., "For this article...").
           - **100% EXACT MATCH**: Every extracted sentence MUST be a strict, exact substring match from the original text.
        5. TITLE RELEVANCY: Only extract facts directly relevant to the main subject, entity, or topic described in the Title. Do NOT extract generic market summaries or boilerplate descriptions of other companies.
        6. NO DUPLICATES: Do NOT extract duplicate or virtually identical facts, even if they appear in slightly different formats (e.g., "$81B" vs "$81 billion"). Copy only the single best instance.
        7. CLUTTER FILTERING: Strictly ignore all promotional text, recommended reading lists, clickbait newsletters, or subscription notices at the end of the article.
        8. CRITICAL: {critical_condition}
        
        Here are the articles to process:
        {{input_text}}"""


# ------------------------------------------------------------------------------
# Dynamic Configuration & Fallback Configuration Loading
# ------------------------------------------------------------------------------
# Define a minimal fallback configuration for open-source default runs.
# This prevents the codebase from failing if the private configs are not present.
DEFAULT_AGENT_CONFIGS = {
    "General": {
        "role": "Exact Text Extraction Algorithm: US Macroeconomics & Federal Policy",
        "goal": "Scan the text and strictly copy-paste exact sentences regarding inflation, Fed rates, fiscal policy, or labor markets.",
        "backstory": build_backstory(
            "US Macroeconomics & Fed Policy", "speculative predictions"
        ),
        "task_description_template": build_task_template(),
        "keywords": {"cpi": 10, "pce": 10, "fed": 5, "inflation": 5},
    },
    "Others": {
        "role": "Exact Text Extraction Algorithm: Miscellaneous Industries",
        "goal": "Scan the text and strictly copy-paste exact sentences regarding general business data, margin changes, and corporate events.",
        "backstory": build_backstory("general corporate data", "CEO personal gossip"),
        "task_description_template": build_task_template(),
        "keywords": {"revenue": 1, "earnings": 1, "margin": 1},
    },
}

# Load the proprietary configurations dynamically if they exist.
# The 'config/' directory is in the .gitignore to protect intellectual property (IP).
AGENT_CONFIGS = DEFAULT_AGENT_CONFIGS

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(
        project_root, "config", "prompts", "extractor_configs.json"
    )

    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            custom_configs = json.load(f)
            if custom_configs and isinstance(custom_configs, dict):
                AGENT_CONFIGS = custom_configs
except Exception:
    # Fail silently and fall back to DEFAULT_AGENT_CONFIGS to maintain engine robustness.
    pass


def get_agent_config(category: str) -> dict:
    """
    Returns the specific prompt configuration for a given category.
    If the category isn't specifically defined in AGENT_CONFIGS,
    it falls back to a default exact-extraction template.
    """
    if category in AGENT_CONFIGS:
        return AGENT_CONFIGS[category]

    # Default fallback for other categories before they are hard-tuned
    return {
        "role": f"{category} Exact Text Extraction Algorithm",
        "goal": f"Your absolute goal is to scan the provided text and strictly COPY AND PASTE the exact original sentences that contain hard data or definitive statements regarding key performance indicators (KPIs) for {category}.",
        "backstory": build_backstory(industry_focus="industry data", ignore_rule=""),
        "task_description_template": build_task_template(),
    }
