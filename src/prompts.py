def build_backstory(industry_focus: str, ignore_rule: str) -> str:
    ignore_section = (
        f"\n        - ALSO IGNORE {ignore_rule}." if ignore_rule else ""
    )
    return f"""You are a deterministic, zero-creativity extraction engine for {industry_focus}.

        [CRITICAL GATING RULE]
        First, check the text's purpose. If the text is a personal finance tutorial, personal retirement calculator guide, credit card/retail mortgage shopping tip, budget-saving tip, or lifestyle column, you MUST immediately output exactly "NO_EXTRACTION" and terminate. Do not extract anything.

        STRICT RULES:
        - NEVER rewrite, summarize, or paraphrase any sentence.
        - ABSOLUTELY IGNORE {ignore_rule}.
        - Your output must be a strict, 100% exact substring match from the source text.
        - NEVER write explanations, formatting tags (like <b>), markdown headers, or intro/outro text. Any output not in the source is an absolute failure."""


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
# 10-Category Prompts & Configuration
# ------------------------------------------------------------------------------
AGENT_CONFIGS = {
    # 1) General (US Macroeconomics & Federal Reserve Policy)
    "General": {
        "role": "Exact Text Extraction Algorithm: US Macroeconomics & Federal Reserve Policy",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Inflation Trajectories: CPI, PCE, Core PCE, and Supercore inflation month-over-month or YoY percentage changes.
        2. Fed Decisions & Projections: FOMC rate pause duration (at the 3.75% upper bound), dot plot voting distributions, and QT limits.
        3. Fiscal Actions & Liquidity: US fiscal deficit ratios, Treasury auction Bid-to-Cover demand ratios, and 2s10s yield curve inversion spreads.
        4. Forward Guidance: Direct quotes from Fed officials and hawkish/dovish semantic pivots.
        5. Structural Labor Shifts: Non-Farm Payrolls (NFP), unemployment rates, labor force participation, and wage growth percentages.
        6. Macro Housing Indicators: National average 30-year fixed mortgage rates, MBA mortgage application indexes, and Case-Shiller index growth. (DO NOT extract individual refinancing tutorials, home shopping guides, or mortgage calculators).
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="US Macroeconomics & Federal Reserve Policy",
            ignore_rule="speculative price predictions, technical analysis (support/resistance), or emotional market sentiment",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "cpi",
            "pce",
            "inflation",
            "fed",
            "federal reserve",
            "interest rate",
            "basis point",
            "bps",
            "dot plot",
            "qt",
            "nfp",
            "payroll",
            "unemployment",
            "hawkish",
            "dovish",
            "yield curve",
            "m2",
            "treasury",
            "macro",
            "gdp",
            "powell",
            "yellen",
            "recession",
            "soft landing",
            "cbdc",
            "debt limit",
            "fiscal",
            "tariff",
            "trade war",
            "fomc",
            "quantitative easing",
            "qe",
            "liquidity",
            "rate cut",
            "rate hike",
            "core pce",
            "treasury yield",
            "pmi",
            "ism",
            "jobless claims",
            "retail sales",
            "consumer sentiment index",
            "university of michigan consumer sentiment",
            "wage growth",
            "sofr",
            "mortgage rate",
            "mortgage applications",
            "housing starts",
            "home sales",
            "case-shiller",
            "delinquency",
            "foreclosure",
        ],
    },
    
    # 2) Bitcoin (Bitcoin & Institutional Liquidity)
    "Bitcoin": {
        "role": "Exact Text Extraction Algorithm: Bitcoin & Institutional Liquidity",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. ETF Flow Dynamics: Daily/weekly net inflows or outflows for Spot BTC and ETH ETFs.
        2. Derivates & Options Volume: Options trading open interest and institutional volume for crypto ETFs.
        3. Corporate Treasury holdings: Public corporate treasury accumulation (e.g., MicroStrategy, Tesla) and premium/discount to Net Asset Value (NAV).
        4. Accounting Rules & Regs: FASB fair value accounting adoption metrics and SEC regulatory approval filings.
        5. Sovereign Strategic Reserves: Nation-state balance sheet purchases and sovereign wealth fund allocations.
        6. Supply & Fundamentals: Halving impacts, structural supply constraints, and global hash rate distribution.
        7. Hard Asset Benchmarks: Gold and Silver price action, ETF flows, and supply/demand dynamics for comparative analysis.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="institutional money flows and on-chain facts",
            ignore_rule="speculative price predictions, technical analysis (support/resistance), or emotional market sentiment",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "etf",
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "inflow",
            "outflow",
            "whale",
            "treasury",
            "fasb",
            "halving",
            "hash rate",
            "tvl",
            "l2",
            "rwa",
            "tokenization",
            "gold",
            "silver",
            "saylor",
            "michael saylor",
            "fink",
            "larry fink",
            "gensler",
            "sec",
            "stablecoin",
            "defi",
            "staking",
            "solana",
            "sol",
            "sovereign fund",
            "strategic reserve",
            "crypto",
            "coinbase",
            "microstrategy",
            "mstr",
            "blackrock",
            "options",
            "liquidation",
            "on-chain",
            "tether",
            "usdt",
            "usdc",
            "miners",
            "mining",
            "supply squeeze",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 3) Semiconductor (Semiconductor & Supply Chain)
    "Semiconductor": {
        "role": "Exact Text Extraction Algorithm: Semiconductor & Supply Chain",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Foundry CapEx & Revenue: TSMC, Samsung, and Intel datacenter/foundry revenue growth rates and capital expenditures.
        2. Lithography Orders: EUV and High-NA EUV system shipment numbers, average selling prices (ASP), and backlog orders.
        3. Advanced Packaging Milestones: CoWoS/SOIC monthly wafer capacity metrics, Glass Substrate yields, and Silicon Photonics adoption percentages.
        4. High Bandwidth Memory: HBM3e/HBM4 product qualifications, shipment allocation percentages, and pricing premiums.
        5. CSP Custom ASICs: In-house ASIC chip tape-out schedules and production volume versus NVIDIA/AMD GPU shipments.
        6. Industry Consolidation & Flows: Semiconductor M&A valuations, public IPO offerings, and venture capital deals.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="semiconductor industry data",
            ignore_rule="speculative price predictions or technical analysis",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "revenue",
            "eps",
            "margin",
            "tsmc",
            "nvidia",
            "amd",
            "cowos",
            "foundry",
            "capex",
            "euv",
            "nm",
            "soic",
            "photonics",
            "chips act",
            "asic",
            "chiplet",
            "terafab",
            "musk",
            "elon musk",
            "jensen huang",
            "lisa su",
            "gelsinger",
            "intel",
            "samsung",
            "hbm",
            "high-na",
            "glass substrate",
            "ai chip",
            "gpu",
            "npu",
            "custom silicon",
            "datacenter",
            "server",
            "arm",
            "broadcom",
            "qualcomm",
            "micron",
            "wafer",
            "packaging",
            "asml",
            "yields",
            "supply chain",
            "fab",
            "texas instruments",
            "nxp",
            "memory",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 4) AI (AI Models & Platforms)
    "AI": {
        "role": "Exact Text Extraction Algorithm: AI & Generative Models",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Model Compute Scales: Training parameters, dataset token sizes, multi-million dollar training run costs, and open-weights release dates.
        2. Inference Cost Reductions: API pricing per million tokens, time-to-first-token (TTFT) latency, and compute cost deflation percentages.
        3. Enterprise ROI Metrics: B2B corporate agent deployment conversion rates, corporate productivity gains (hours saved), and software API volumes.
        4. Edge AI Specs: Local execution requirements for AI PCs/Smartphones (e.g., NPU TOPS, minimum unified RAM gigabytes).
        5. Cluster Infrastructure: Hyper-scalers AI cluster node counts (e.g., number of GPUs in a single cluster like Colossus) and networking bandwidth speeds.
        6. Startup Deal flows: High-profile AI startup funding rounds (e.g., OpenAI, Anthropic, xAI), private valuations, and AI IPO registrations.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="AI industry data",
            ignore_rule="ethical debates, generic future outlooks, or technical analysis",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "compute",
            "vram",
            "gpu",
            "api",
            "b2b",
            "funding",
            "valuation",
            "nuclear",
            "smr",
            "roi",
            "agi",
            "multi-modal",
            "edge ai",
            "agent",
            "autonomous",
            "sam altman",
            "openai",
            "anthropic",
            "demis hassabis",
            "ilya sutskever",
            "xai",
            "grok",
            "llama",
            "claude",
            "gemini",
            "copilot",
            "chatgpt",
            "deepseek",
            "qwen",
            "moe",
            "generative ai",
            "llm",
            "large language model",
            "inference",
            "training",
            "ai agent",
            "data center",
            "datacenter",
            "power cooling",
            "liquid cooling",
            "open source",
            "parameters",
            "scaling law",
            "ipo",
            "m&a",
            "acquisition",
        ],
    },
    
    # 5) Power & Grid (Power Infrastructure & Energy)
    "Power & Grid": {
        "role": "Exact Text Extraction Algorithm: Power Infrastructure & Energy",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Power Purchase Agreements: Megawatts (MW) or Gigawatts (GW) of energy secured under contract by tech companies, and contract pricing per megawatt-hour ($/MWh).
        2. Clean Energy & Nuclear Capacity: Nuclear plant power allocations, Small Modular Reactor (SMR) development schedules, and uranium fuel supply metrics.
        3. Grid Hardware lead times: Transformer lead times (months/weeks), switchgear backlogs, and utility substation infrastructure CapEx.
        4. Utility-Scale Battery Storage: Energy Storage System (ESS) capacity installations (MWh), and battery storage discharge duration metrics.
        5. Utility Infrastructure Consolidation: Energy sector M&A deals, utility IPOs, and capital flows for power grid upgrades.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="power grid, utilities, and energy infrastructure data",
            ignore_rule="general climate changes or retail energy bills",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "power purchase agreement",
            "ppa",
            "nuclear",
            "smr",
            "uranium",
            "transformer",
            "switchgear",
            "substation",
            "ess",
            "energy storage",
            "battery storage",
            "utility",
            "utilities",
            "ge vernova",
            "constellation energy",
            "vistra",
            "nrg energy",
            "clean energy",
            "electricity",
            "power grid",
            "grid interconnect",
            "megawatts",
            "gigawatts",
            "mwh",
            "gwh",
            "eaton",
            "powell industries",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 6) Robotics & Autonomy (Robotics & Autonomous Systems)
    "Robotics & Autonomy": {
        "role": "Exact Text Extraction Algorithm: Robotics & Autonomous Systems",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Autonomous Driving Miles: Cumulative FSD (Full Self-Driving) miles driven, average miles between disengagements, and safety performance comparisons.
        2. Robotaxi Fleet Operations: Operating Robotaxi vehicle counts, daily active autonomous rides, and regulatory permitting approvals.
        3. Humanoid Robot Pilots: Factory floor humanoid deployments, unit manufacturing costs, payload capacities, and hand dexterity degrees of freedom.
        4. Logistics & Warehousing Automation: Automated guided vehicle (AGV) orders, robotic warehouse system backlogs (e.g., Symbotic metrics), and throughput efficiency gains.
        5. EV Batteries & Supply Chain: Next-gen battery cell costs ($/kWh), solid-state battery development, and localization metrics.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="robotics, autonomous systems, and advanced mobility hardware",
            ignore_rule="speculative consumer product release hype or personal stock tips",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "fsd",
            "full self-driving",
            "robotaxi",
            "humanoid",
            "humanoid robot",
            "optimus",
            "figure ai",
            "agv",
            "automated guided vehicle",
            "symbotic",
            "warehouse automation",
            "industrial automation",
            "tesla",
            "tsla",
            "battery cell",
            "solid-state battery",
            "boston dynamics",
            "rockwell automation",
            "robot",
            "robotics",
            "autonomy",
            "autonomous driving",
            "disengagement",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 7) Software (Enterprise Software & Cloud Services)
    "Software": {
        "role": "Exact Text Extraction Algorithm: Enterprise Software & Cloud Services",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Consumption Monetization: Revenue percentages from consumption-based or outcome-based billing vs. traditional seat-based licensing.
        2. Remaining Performance Obligations: Total RPO contract values, NRR (Net Retention Rate) percentages, and NDR (Net Dollar Retention) rates.
        3. SecOps Automation Budgets: Cloud security automation spend, platform consolidation numbers, and active threat detection rates.
        4. SaaS Churn & Adoption: Enterprise software user seat churn rates, agentic SaaS seat replacement metrics, and software seat pricing changes.
        5. Enterprise Software M&A/IPOs: Software company mergers, valuation multiples of acquisitions, and SaaS IPO pipeline status.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="software industry data",
            ignore_rule="generic feature updates or UI changes",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "aws",
            "azure",
            "google cloud",
            "gcp",
            "nrr",
            "ndr",
            "copilot",
            "arpu",
            "rpo",
            "saas",
            "cybersecurity",
            "nadella",
            "jassy",
            "pichai",
            "benioff",
            "salesforce",
            "servicenow",
            "crowdstrike",
            "palo alto",
            "databricks",
            "snowflake",
            "cloud computing",
            "enterprise",
            "b2b saas",
            "arr",
            "mrr",
            "churn",
            "adoption",
            "cisa",
            "zero trust",
            "endpoint security",
            "database",
            "oracle",
            "sap",
            "adobe",
            "workday",
            "palantir",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 8) Bio (Biotechnology & Pharmaceuticals)
    "Bio": {
        "role": "Exact Text Extraction Algorithm: Biotechnology & Pharmaceuticals",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Trial Efficacy & Stats: Phase 1/2/3 clinical trial primary endpoint efficacy percentages, patient weight loss percentages, and p-values.
        2. Regulatory Decisions: FDA PDUFA action dates, CRL (Complete Response Letter) rejections, and European EMA approvals.
        3. GLP-1 Indication Expansions: Cardiovascular event risk reductions, kidney disease delay metrics, and sleep apnea trial data.
        4. Obesity pipeline formulations: Oral GLP-1 bioavailability percentages, muscle-mass preservation ratios, and dosing schedules.
        5. CDMO capacity expansions: Vial filling line throughput capacities, CDMO contract backlogs, and syringe supply metrics.
        6. BioTech Consolidation: Biotechnology mergers & acquisitions (M&A), major VC funding rounds, and clinical-stage BioTech IPOs.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="biotechnology industry data",
            ignore_rule="generic health advice or stock price predictions",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "clinical",
            "phase 1",
            "phase 2",
            "phase 3",
            "p-value",
            "fda",
            "pdufa",
            "crl",
            "cmo",
            "glp-1",
            "obesity",
            "patent",
            "ira",
            "m&a",
            "crispr",
            "alphafold",
            "eli lilly",
            "novo nordisk",
            "wegovy",
            "zepbound",
            "mrna",
            "car-t",
            "gene editing",
            "alzheimer",
            "oncology",
            "trial",
            "efficacy",
            "safety",
            "cro",
            "cdmo",
            "weight loss",
            "pipeline",
            "biotech",
            "pharma",
            "blockbuster",
            "fda approval",
            "immunotherapy",
            "biosimilar",
            "orphan drug",
            "ipo",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 9) Aerospace (Aerospace, Space Economy & Defense)
    "Aerospace": {
        "role": "Exact Text Extraction Algorithm: Aerospace & Space Economy",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Launch Cost & Freq: Cost per launch payload kilogram ($/kg), rocket booster reuse cycles, and annual/monthly launch counts.
        2. Satellite Broadband (Starlink): Active subscribers, monthly average revenue per user (ARPU), and total satellite constellations.
        3. Autonomous Defense Systems: DoD contract award values for AI-driven drone swarms, tactical hypersonic speed metrics, and hardware quantities.
        4. Military Software Backlogs: Palantir and Anduril multi-year contract values and government defense budget allocations.
        5. Commercial aviation backlogs: Boeing and Airbus commercial jet order backlog years, monthly production rates, and delivery delays.
        6. Defense Consolidation: Aerospace & defense mergers & acquisitions (M&A), defense contractor IPOs, and venture funding for defense tech startups.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="aerospace industry data",
            ignore_rule="generic space exploration enthusiasm or stock price predictions",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "launch",
            "payload",
            "starlink",
            "nasa",
            "dod",
            "tam",
            "defense",
            "backlog",
            "orbit",
            "drone",
            "hypersonic",
            "artemis",
            "lunar",
            "spacex",
            "blue origin",
            "ula",
            "lockheed",
            "boeing",
            "northrop",
            "anduril",
            "palantir",
            "karp",
            "shield ai",
            "space",
            "satellite",
            "leo",
            "rocket",
            "missile",
            "rtx",
            "raytheon",
            "general dynamics",
            "defense budget",
            "pentagon",
            "procurement",
            "aviation",
            "aero",
            "military",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    
    # 10) Others (Big Tech Platforms, Consumer, Global Supply Chains & Valuation)
    "Others": {
        "role": "Exact Text Extraction Algorithm: Mega-Cap Tech & Consumer Platforms",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Big Tech Platform Revenues: Advertising revenue growth for Meta/Google, ad impression CPM/CPC rate changes, social active users (DAU/MAU), and Apple iPhone unit shipments and Services segment gross margins.
        2. E-Commerce & Retail: Amazon retail GMV growth, Shopify GMV, same-store sales (SSS) growth, and operating margins for Walmart, Costco, or Target.
        3. Consumer Credit & Macro Consumption: Credit card delinquency rates, auto loan default rates, and overall retail sales percentage changes.
        4. Logistics & Supply Chain: Same-day delivery metropolitan coverage, inventory turn ratios, and reshoring/nearshoring relocation metrics (e.g., China+1).
        5. Shareholder Capital Returns: Free Cash Flow (FCF) yields, share buyback sizes, and dividend distribution plans.
        6. Corporate Governance & Regulations: Major antitrust trial rulings, regulatory fines, and shareholder activist voting metrics.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="mega-cap and consumer industry data",
            ignore_rule="speculative price predictions, emotional sentiment, or CEO personal gossip",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "delivery",
            "shipment",
            "fcf",
            "free cash flow",
            "margin",
            "logistics",
            "supply chain",
            "reshoring",
            "robotics",
            "automation",
            "buyback",
            "dividend",
            "robotaxi",
            "fsd",
            "tim cook",
            "zuckerberg",
            "bezos",
            "tesla",
            "apple",
            "meta",
            "amazon",
            "alphabet",
            "optimus",
            "figure ai",
            "consumer spending",
            "retail",
            "ecommerce",
            "ev",
            "electric vehicle",
            "iphone",
            "services revenue",
            "ad revenue",
            "advertising",
            "capex",
            "operating income",
            "cost cutting",
            "layoffs",
            "shareholder return",
            "walmart",
            "costco",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
            "anti-trust",
            "caterpillar",
            "genset",
            "gensets",
            "turbine",
            "reciprocating engine",
            "power grid",
            "electricity",
            "utility",
            "utilities",
            "ge vernova",
            "constellation energy",
            "google",
            "goog",
            "googl",
            "netflix",
            "nflx",
            "target",
            "tgt",
            "shopify",
            "shop",
            "mercadolibre",
            "meli",
            "china+1",
            "nearshoring",
            "antitrust",
        ],
    },
}


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
